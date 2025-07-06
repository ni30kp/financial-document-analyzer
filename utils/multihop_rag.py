"""
Multi-hop RAG for complex queries
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
            
            logger.info(f"Decomposed into {len(sub_queries)} sub-questions")
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
            enhanced_query = f"Given context: {previous_context[:500]}\n\nAnswer: {sub_query}"
        
        context = self.rag_pipeline.retrieve_context(enhanced_query, top_k=5)
        
        reasoning_prompt = f"""
        Based on the context, answer this question: {sub_query}

        Context:
        {context}

        Format:
        ANSWER: [your answer]
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
                retrieved_context="",
                reasoning="Error processing query",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def synthesize_final_answer(self, query: str, hops: List[HopResult]) -> Tuple[str, str, float]:
        """Synthesize final answer from all hops"""
        
        hop_summaries = []
        for hop in hops:
            hop_summaries.append(f"Q: {hop.sub_query}\nA: {hop.reasoning}")
        
        synthesis_prompt = f"""
        Original question: {query}

        Findings:
        {chr(10).join(hop_summaries)}

        Provide a comprehensive answer to the original question.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.ULTRASAFE_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=800,
                temperature=0.1
            )
            
            final_answer = response.choices[0].message.content.strip()
            
            # Create reasoning chain
            reasoning_chain = " -> ".join([hop.sub_query for hop in hops])
            
            # Calculate overall confidence
            avg_confidence = sum(hop.confidence for hop in hops) / len(hops) if hops else 0.0
            
            return final_answer, reasoning_chain, avg_confidence
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return "Error generating final answer", "", 0.0

class MultiHopRAGPipeline:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.decomposer = QueryDecomposer()
        self.reasoning_engine = ReasoningEngine(rag_pipeline)
        
    def process_query(self, query: str) -> MultiHopResult:
        """Process a complex query using multi-hop reasoning"""
        start_time = datetime.now()
        
        # Classify and decompose query
        query_type = self.decomposer.classify_query(query)
        sub_queries = self.decomposer.decompose_query(query, query_type)
        
        # Execute hops
        hops = []
        previous_context = ""
        
        for i, sub_query in enumerate(sub_queries):
            hop = self.reasoning_engine.execute_hop(sub_query, i + 1, previous_context)
            hops.append(hop)
            previous_context = hop.reasoning
        
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
        """Check if query requires multi-hop reasoning"""
        return self.decomposer.classify_query(query) != QueryType.SIMPLE
    
    def get_query_preview(self, query: str) -> Dict[str, Any]:
        """Get a preview of how the query would be processed"""
        query_type = self.decomposer.classify_query(query)
        sub_queries = self.decomposer.decompose_query(query, query_type)
        
        return {
            "query_type": query_type.value,
            "sub_queries": sub_queries,
            "estimated_hops": len(sub_queries),
            "is_complex": query_type != QueryType.SIMPLE
        } 