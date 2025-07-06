"""
Multi-Hop RAG Pipeline for Financial Document Analysis
"""

import logging
import json
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from .rag_pipeline import RAGPipeline
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    SIMPLE = "simple"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend"
    CAUSAL = "causal"
    SYNTHESIS = "synthesis"
    COMPLEX = "complex"

@dataclass
class HopResult:
    hop_number: int
    sub_query: str
    retrieved_context: str
    reasoning: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class MultiHopResult:
    original_query: str
    query_type: QueryType
    hops: List[HopResult]
    final_answer: str
    reasoning_chain: str
    confidence: float
    total_hops: int
    execution_time: float

class QueryDecomposer:
    def __init__(self):
        self.settings = Settings()
        self.client = OpenAI(
            api_key=self.settings.ULTRASAFE_API_KEY,
            base_url=self.settings.ULTRASAFE_BASE_URL
        )
        
    def classify_query(self, query: str) -> QueryType:
        query_lower = query.lower()
        
        comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'better', 'worse']
        trend_keywords = ['trend', 'over time', 'growth', 'decline', 'change', 'evolution']
        causal_keywords = ['why', 'because', 'reason', 'cause', 'effect', 'impact', 'influence']
        synthesis_keywords = ['overall', 'summary', 'comprehensive', 'analysis', 'assessment']
        
        if any(keyword in query_lower for keyword in comparison_keywords):
            return QueryType.COMPARISON
        elif any(keyword in query_lower for keyword in trend_keywords):
            return QueryType.TREND_ANALYSIS
        elif any(keyword in query_lower for keyword in causal_keywords):
            return QueryType.CAUSAL
        elif any(keyword in query_lower for keyword in synthesis_keywords):
            return QueryType.SYNTHESIS
        elif len(query.split()) > 15 or '?' in query and len(query.split('?')) > 1:
            return QueryType.COMPLEX
        else:
            return QueryType.SIMPLE
    
    def decompose_query(self, query: str, query_type: QueryType) -> List[str]:
        if query_type == QueryType.SIMPLE:
            return [query]
        
        decomposition_prompt = f"""
        Break down this {query_type.value} query into 2-4 specific sub-questions:

        Query: {query}

        Requirements:
        1. Each sub-question should be specific and focused
        2. Sub-questions should build on each other logically
        3. Cover all aspects of the original query
        4. Be suitable for document retrieval

        Return only the sub-questions as a JSON array of strings.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.ULTRASAFE_MODEL,
                messages=[{"role": "user", "content": decomposition_prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith('[') and content.endswith(']'):
                sub_queries = json.loads(content)
            else:
                json_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                if json_match:
                    sub_queries = json.loads(json_match.group(0))
                else:
                    lines = content.split('\n')
                    sub_queries = []
                    for line in lines:
                        if re.match(r'^\d+\.', line.strip()):
                            sub_queries.append(line.strip().split('.', 1)[1].strip())
            
            logger.info(f"Decomposed query into {len(sub_queries)} sub-questions")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return self._simple_decompose(query, query_type)
    
    def _simple_decompose(self, query: str, query_type: QueryType) -> List[str]:
        if query_type == QueryType.COMPARISON:
            return [
                f"What are the key metrics mentioned in the query: {query}?",
                f"What are the specific values for comparison in: {query}?",
                f"What factors explain the differences in: {query}?"
            ]
        elif query_type == QueryType.TREND_ANALYSIS:
            return [
                f"What are the historical values for: {query}?",
                f"What are the current values for: {query}?",
                f"What factors drove the changes in: {query}?"
            ]
        else:
            return [query]

class ReasoningEngine:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.settings = Settings()
        self.client = OpenAI(
            api_key=self.settings.ULTRASAFE_API_KEY,
            base_url=self.settings.ULTRASAFE_BASE_URL
        )
        self.max_hops = 5
        
    def execute_hop(self, sub_query: str, hop_number: int, previous_context: str = "") -> HopResult:
        enhanced_query = sub_query
        if previous_context:
            enhanced_query = f"Given the previous context: {previous_context[:500]}...\n\nNow answer: {sub_query}"
        
        context = self.rag_pipeline.retrieve_context(enhanced_query, top_k=5)
        
        reasoning_prompt = f"""
        Based on the retrieved context, provide a focused answer to this sub-question: {sub_query}

        Context:
        {context}

        Provide your answer in this format:
        ANSWER: [your specific answer]
        REASONING: [your reasoning]
        CONFIDENCE: [0.0-1.0]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.ULTRASAFE_MODEL,
                messages=[{"role": "user", "content": reasoning_prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract answer and reasoning
            answer_match = re.search(r'ANSWER:\s*(.*?)(?=REASONING:|$)', content, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=CONFIDENCE:|$)', content, re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
            
            answer = answer_match.group(1).strip() if answer_match else content
            reasoning = reasoning_match.group(1).strip() if reasoning_match else content
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            return HopResult(
                hop_number=hop_number,
                sub_query=sub_query,
                retrieved_context=context,
                reasoning=reasoning,
                confidence=confidence,
                metadata={"enhanced_query": enhanced_query}
            )
            
        except Exception as e:
            logger.error(f"Error in hop {hop_number}: {e}")
            return HopResult(
                hop_number=hop_number,
                sub_query=sub_query,
                retrieved_context=context,
                reasoning=f"Error processing query: {e}",
                confidence=0.1,
                metadata={"error": str(e)}
            )
    
    def synthesize_final_answer(self, query: str, hops: List[HopResult]) -> Tuple[str, str, float]:
        hop_summaries = []
        for hop in hops:
            hop_summaries.append(f"Q{hop.hop_number}: {hop.sub_query}\nA{hop.hop_number}: {hop.reasoning}")
        
        synthesis_prompt = f"""
        Original Question: {query}
        
        Sub-question Analysis:
        {chr(10).join(hop_summaries)}
        
        Based on the analysis above, provide a comprehensive final answer to the original question.
        
        Your response should:
        1. Synthesize insights from all sub-questions
        2. Provide a clear, complete answer
        3. Include specific details and evidence
        4. Be well-structured and coherent
        
        Final Answer:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.ULTRASAFE_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=800,
                temperature=0.2
            )
            
            final_answer = response.choices[0].message.content.strip()
            
            # Create reasoning chain
            reasoning_chain = f"Multi-hop analysis with {len(hops)} steps:\n"
            for i, hop in enumerate(hops, 1):
                reasoning_chain += f"{i}. {hop.sub_query} -> {hop.reasoning}\n"
            reasoning_chain += f"Final synthesis: {final_answer}"
            
            # Calculate overall confidence
            confidence = sum(hop.confidence for hop in hops) / len(hops)
            
            return final_answer, reasoning_chain, confidence
            
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {e}")
            
            # Fallback answer
            fallback_answer = f"Based on the multi-hop analysis, here are the key findings:\n"
            for hop in hops:
                fallback_answer += f"- {hop.reasoning}\n"
            
            return fallback_answer, "Error in synthesis", 0.5

class MultiHopRAGPipeline:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.query_decomposer = QueryDecomposer()
        self.reasoning_engine = ReasoningEngine(rag_pipeline)
        
    def process_query(self, query: str) -> MultiHopResult:
        start_time = datetime.now()
        
        # Classify query
        query_type = self.query_decomposer.classify_query(query)
        
        # Decompose query
        sub_queries = self.query_decomposer.decompose_query(query, query_type)
        
        # Execute hops
        hops = []
        previous_context = ""
        
        for i, sub_query in enumerate(sub_queries, 1):
            hop_result = self.reasoning_engine.execute_hop(sub_query, i, previous_context)
            hops.append(hop_result)
            previous_context += f" {hop_result.reasoning}"
        
        # Synthesize final answer
        final_answer, reasoning_chain, confidence = self.reasoning_engine.synthesize_final_answer(query, hops)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MultiHopResult(
            original_query=query,
            query_type=query_type,
            hops=hops,
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            total_hops=len(hops),
            execution_time=execution_time
        )
    
    def is_complex_query(self, query: str) -> bool:
        query_type = self.query_decomposer.classify_query(query)
        return query_type != QueryType.SIMPLE
    
    def get_query_preview(self, query: str) -> Dict[str, Any]:
        query_type = self.query_decomposer.classify_query(query)
        requires_multihop = query_type != QueryType.SIMPLE
        
        preview = {
            "query_type": query_type.value,
            "requires_multihop": requires_multihop,
            "estimated_hops": 1 if not requires_multihop else 3,
            "sub_queries": []
        }
        
        if requires_multihop:
            try:
                sub_queries = self.query_decomposer.decompose_query(query, query_type)
                preview["sub_queries"] = sub_queries
                preview["estimated_hops"] = len(sub_queries)
            except Exception as e:
                logger.error(f"Error generating preview: {e}")
                preview["sub_queries"] = [query]
        
        return preview 