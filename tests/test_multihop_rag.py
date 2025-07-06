"""
Test cases for Multi-Hop RAG functionality
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime

from utils.multihop_rag import (
    QueryDecomposer, 
    ReasoningEngine, 
    MultiHopRAGPipeline,
    QueryType,
    HopResult,
    MultiHopResult
)
from utils.rag_pipeline import RAGPipeline
from config.settings import Settings

class TestQueryDecomposer:
    """Test query decomposition functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.decomposer = QueryDecomposer()
    
    def test_classify_simple_query(self):
        """Test classification of simple queries"""
        query = "What was the revenue last quarter?"
        query_type = self.decomposer.classify_query(query)
        assert query_type == QueryType.SIMPLE
    
    def test_classify_comparison_query(self):
        """Test classification of comparison queries"""
        query = "Compare revenue growth between Q1 and Q4"
        query_type = self.decomposer.classify_query(query)
        assert query_type == QueryType.COMPARISON
    
    def test_classify_trend_query(self):
        """Test classification of trend analysis queries"""
        query = "How has revenue changed over time?"
        query_type = self.decomposer.classify_query(query)
        assert query_type == QueryType.TREND_ANALYSIS
    
    def test_classify_causal_query(self):
        """Test classification of causal queries"""
        query = "Why did profits decline and what caused the impact?"
        query_type = self.decomposer.classify_query(query)
        assert query_type == QueryType.CAUSAL
    
    def test_classify_complex_query(self):
        """Test classification of complex queries"""
        query = "What are the comprehensive financial metrics and how do they relate to market conditions and strategic decisions?"
        query_type = self.decomposer.classify_query(query)
        assert query_type == QueryType.COMPLEX
    
    def test_simple_query_decomposition(self):
        """Test that simple queries are not decomposed"""
        query = "What was the revenue?"
        query_type = QueryType.SIMPLE
        sub_queries = self.decomposer.decompose_query(query, query_type)
        assert len(sub_queries) == 1
        assert sub_queries[0] == query
    
    @patch('utils.multihop_rag.OpenAI')
    def test_comparison_query_decomposition(self, mock_openai):
        """Test decomposition of comparison queries"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''[
            "What was the revenue in Q1?",
            "What was the revenue in Q4?",
            "What factors contributed to the revenue difference?"
        ]'''
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = "Compare revenue between Q1 and Q4"
        query_type = QueryType.COMPARISON
        sub_queries = self.decomposer.decompose_query(query, query_type)
        
        assert len(sub_queries) == 3
        assert "Q1" in sub_queries[0]
        assert "Q4" in sub_queries[1]
        assert "factors" in sub_queries[2].lower()
    
    def test_fallback_decomposition(self):
        """Test fallback decomposition when API fails"""
        query = "Compare revenue trends between companies"
        query_type = QueryType.COMPARISON
        
        # This should use fallback decomposition
        sub_queries = self.decomposer._simple_decompose(query, query_type)
        
        assert len(sub_queries) == 3
        assert any("key metrics" in sq for sq in sub_queries)
        assert any("specific values" in sq for sq in sub_queries)
        assert any("factors explain" in sq for sq in sub_queries)

class TestReasoningEngine:
    """Test reasoning engine functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_rag_pipeline = Mock()
        self.reasoning_engine = ReasoningEngine(self.mock_rag_pipeline)
    
    def test_extract_section(self):
        """Test section extraction from formatted responses"""
        content = """
        ANSWER: The revenue increased by 15%
        CONFIDENCE: 0.85
        INSIGHTS: Strong growth in Q4
        """
        
        answer = self.reasoning_engine._extract_section(content, "ANSWER")
        confidence_text = self.reasoning_engine._extract_section(content, "CONFIDENCE")
        insights = self.reasoning_engine._extract_section(content, "INSIGHTS")
        
        assert "revenue increased by 15%" in answer
        assert "0.85" in confidence_text
        assert "Strong growth" in insights
    
    def test_extract_confidence(self):
        """Test confidence extraction"""
        content = "CONFIDENCE: 0.75"
        confidence = self.reasoning_engine._extract_confidence(content)
        assert confidence == 0.75
        
        # Test fallback
        content = "No confidence mentioned"
        confidence = self.reasoning_engine._extract_confidence(content)
        assert confidence == 0.5
    
    @patch('utils.multihop_rag.OpenAI')
    def test_execute_hop(self, mock_openai):
        """Test single hop execution"""
        # Mock RAG pipeline
        self.mock_rag_pipeline.retrieve_context.return_value = "Revenue was $100M in Q1"
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        ANSWER: The revenue in Q1 was $100M
        CONFIDENCE: 0.9
        INSIGHTS: Revenue showed strong performance
        """
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Execute hop
        result = self.reasoning_engine.execute_hop("What was Q1 revenue?", 1)
        
        assert result.hop_number == 1
        assert result.sub_query == "What was Q1 revenue?"
        assert "$100M" in result.reasoning
        assert result.confidence == 0.9
        assert "Revenue showed strong performance" in result.metadata["insights"]
    
    @patch('utils.multihop_rag.OpenAI')
    def test_synthesize_final_answer(self, mock_openai):
        """Test final answer synthesis"""
        # Create mock hops
        hops = [
            HopResult(
                hop_number=1,
                sub_query="What was Q1 revenue?",
                retrieved_context="Q1 revenue was $100M",
                reasoning="Revenue in Q1 was $100M",
                confidence=0.9,
                metadata={}
            ),
            HopResult(
                hop_number=2,
                sub_query="What was Q4 revenue?",
                retrieved_context="Q4 revenue was $120M",
                reasoning="Revenue in Q4 was $120M",
                confidence=0.85,
                metadata={}
            )
        ]
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        FINAL_ANSWER: Revenue grew from $100M in Q1 to $120M in Q4, representing 20% growth
        REASONING_CHAIN: First found Q1 revenue, then Q4 revenue, then calculated growth
        CONFIDENCE: 0.87
        """
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Synthesize
        final_answer, reasoning_chain, confidence = self.reasoning_engine.synthesize_final_answer(
            "Compare Q1 and Q4 revenue", hops
        )
        
        assert "20% growth" in final_answer
        assert "First found Q1" in reasoning_chain
        assert confidence == 0.87

class TestMultiHopRAGPipeline:
    """Test complete multi-hop RAG pipeline"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_rag_pipeline = Mock()
        self.multihop_rag = MultiHopRAGPipeline(self.mock_rag_pipeline)
    
    def test_is_complex_query(self):
        """Test complex query detection"""
        simple_query = "What was the revenue?"
        complex_query = "Compare revenue trends and analyze the factors driving growth"
        
        assert not self.multihop_rag.is_complex_query(simple_query)
        assert self.multihop_rag.is_complex_query(complex_query)
    
    def test_get_query_preview(self):
        """Test query preview functionality"""
        query = "Compare revenue between Q1 and Q4"
        preview = self.multihop_rag.get_query_preview(query)
        
        assert preview['query_type'] == 'comparison'
        assert preview['requires_multihop'] == True
        assert preview['estimated_hops'] > 1
        assert len(preview['sub_queries']) > 1
    
    @patch('utils.multihop_rag.OpenAI')
    def test_process_query_integration(self, mock_openai):
        """Test end-to-end query processing"""
        # Mock RAG pipeline
        self.mock_rag_pipeline.retrieve_context.return_value = "Revenue data: Q1 $100M, Q4 $120M"
        
        # Mock OpenAI responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        
        # Mock decomposition response
        mock_response.choices[0].message.content = '''[
            "What was the revenue in Q1?",
            "What was the revenue in Q4?"
        ]'''
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Process query
        query = "Compare revenue between Q1 and Q4"
        result = self.multihop_rag.process_query(query)
        
        assert isinstance(result, MultiHopResult)
        assert result.original_query == query
        assert result.query_type == QueryType.COMPARISON
        assert result.total_hops > 0
        assert result.execution_time > 0
        assert result.confidence >= 0.0 and result.confidence <= 1.0

class TestIntegrationWithFinancialSystem:
    """Integration tests with the financial system"""
    
    def setup_method(self):
        """Set up integration test environment"""
        # These tests require the actual system to be initialized
        # They should be run with a working environment
        pass
    
    @pytest.mark.integration
    def test_multihop_with_real_documents(self):
        """Test multi-hop RAG with real financial documents"""
        # This test requires actual documents and API access
        # It should be run in a real environment with proper setup
        pass
    
    @pytest.mark.integration
    def test_streamlit_integration(self):
        """Test integration with Streamlit app"""
        # This test would verify the Streamlit app works with multi-hop RAG
        # It requires a running Streamlit environment
        pass

# Test data and fixtures
@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing"""
    return {
        "revenue_q1": 100000000,
        "revenue_q4": 120000000,
        "profit_q1": 15000000,
        "profit_q4": 20000000,
        "growth_rate": 0.20,
        "factors": ["market expansion", "new product launch", "cost optimization"]
    }

@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return {
        "simple": "What was the revenue?",
        "comparison": "Compare Q1 and Q4 revenue performance",
        "trend": "How has profitability changed over time?",
        "causal": "Why did revenue increase and what factors contributed?",
        "complex": "Analyze the comprehensive financial performance including revenue trends, profitability factors, and strategic implications"
    }

# Example usage and demonstration
if __name__ == "__main__":
    # Run basic tests
    print("Running Multi-Hop RAG tests...")
    
    # Test query classification
    decomposer = QueryDecomposer()
    
    test_queries = [
        "What was the revenue?",
        "Compare revenue between Q1 and Q4",
        "How has profitability changed over time?",
        "Why did costs increase and what caused the impact?",
        "Provide a comprehensive analysis of financial performance and strategic implications"
    ]
    
    for query in test_queries:
        query_type = decomposer.classify_query(query)
        print(f"Query: {query}")
        print(f"Type: {query_type.value}")
        print("---")
    
    print("Multi-Hop RAG tests completed!") 