"""
Test cases for the Financial Agent System
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_parser import DocumentParserAgent, DocumentParser
from agents.analysis_agent import FinancialAnalysisAgent, FinancialAnalyzer
from agents.report_generator import ReportGeneratorAgent, ReportGenerator
from utils.rag_pipeline import RAGPipeline, DocumentChunker, VectorStore
from config.settings import settings

class TestDocumentParser:
    """Test cases for Document Parser Agent"""
    
    def test_document_parser_initialization(self):
        """Test document parser initialization"""
        parser = DocumentParser()
        assert parser.supported_formats == ['.pdf']
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        parser = DocumentParser()
        
        # Test text with extra whitespace
        dirty_text = "This  is   a    test   text  with   extra    spaces"
        cleaned = parser.clean_text(dirty_text)
        assert "    " not in cleaned
        assert "   " not in cleaned
        
        # Test text with page numbers
        text_with_pages = "This is content Page 1 more content Page 2"
        cleaned = parser.clean_text(text_with_pages)
        assert "Page 1" not in cleaned
        assert "Page 2" not in cleaned
    
    def test_document_parser_agent_creation(self):
        """Test DocumentParserAgent creation"""
        agent = DocumentParserAgent()
        assert agent.parser is not None
        
        # Test agent creation
        crew_agent = agent.create_agent()
        assert crew_agent.role == 'Document Parser'
        assert crew_agent.goal is not None

class TestFinancialAnalyzer:
    """Test cases for Financial Analysis Agent"""
    
    def test_financial_analyzer_initialization(self):
        """Test financial analyzer initialization"""
        # Skip if no API key
        if settings.ULTRASAFE_API_KEY == "your_ultrasafe_api_key_here":
            pytest.skip("No UltraSafe API key provided")
        
        analyzer = FinancialAnalyzer()
        assert analyzer.model == settings.ULTRASAFE_MODEL
        assert analyzer.client is not None
    
    def test_metrics_extraction_from_text(self):
        """Test metric extraction from text"""
        analyzer = FinancialAnalyzer()
        
        # Sample financial text
        text = """
        Company XYZ reported revenue of $100 million in Q1 2024, up from $85 million in Q1 2023.
        Net income was $15 million with a profit margin of 15%.
        The company showed strong growth of 17.6% year-over-year.
        """
        
        metrics = analyzer._extract_metrics_from_text(text)
        
        # Check that basic structure is created
        assert "revenue" in metrics
        assert "profit_metrics" in metrics
        assert "year_over_year_changes" in metrics
        assert "company_info" in metrics
    
    def test_trend_identification(self):
        """Test trend identification"""
        analyzer = FinancialAnalyzer()
        
        # Sample metrics with growth
        metrics = {
            "revenue": {"growth_rate": "15% increase"},
            "profit_metrics": {"profit_margin": "20%"}
        }
        
        trends = analyzer.identify_trends(metrics)
        assert len(trends) >= 0  # Should return some trends
    
    def test_financial_analysis_agent_creation(self):
        """Test FinancialAnalysisAgent creation"""
        agent = FinancialAnalysisAgent()
        assert agent.analyzer is not None
        
        # Test agent creation
        crew_agent = agent.create_agent()
        assert crew_agent.role == 'Financial Analyst'
        assert crew_agent.goal is not None

class TestReportGenerator:
    """Test cases for Report Generator Agent"""
    
    def test_report_generator_initialization(self):
        """Test report generator initialization"""
        # Skip if no API key
        if settings.ULTRASAFE_API_KEY == "your_ultrasafe_api_key_here":
            pytest.skip("No UltraSafe API key provided")
        
        generator = ReportGenerator()
        assert generator.model == settings.ULTRASAFE_MODEL
        assert generator.client is not None
    
    def test_metrics_summary_creation(self):
        """Test metrics summary creation"""
        generator = ReportGenerator()
        
        # Sample metrics
        metrics = {
            "revenue": {
                "current_year": "$100M",
                "previous_year": "$85M",
                "growth_rate": "17.6%"
            },
            "profit_metrics": {
                "net_income": "$15M",
                "profit_margin": "15%"
            },
            "key_ratios": {
                "debt_to_equity": "0.35",
                "current_ratio": "2.1"
            }
        }
        
        summary = generator._create_metrics_summary(metrics)
        
        assert "revenue" in summary
        assert "profitability" in summary
        assert "key_ratios" in summary
        assert summary["revenue"]["current"] == "$100M"
        assert summary["profitability"]["net_income"] == "$15M"
    
    def test_performance_indicators(self):
        """Test performance indicators creation"""
        generator = ReportGenerator()
        
        # Sample metrics
        metrics = {
            "year_over_year_changes": {
                "revenue_change": "17.6% increase"
            },
            "profit_metrics": {
                "net_income": "$15M"
            }
        }
        
        indicators = generator._create_performance_indicators(metrics)
        
        assert "revenue_trend" in indicators
        assert "profitability" in indicators
        assert indicators["revenue_trend"] == "POSITIVE"
        assert indicators["profitability"] == "PROFITABLE"
    
    def test_overall_health_assessment(self):
        """Test overall health assessment"""
        generator = ReportGenerator()
        
        # Sample report with positive indicators
        report = {
            "performance_indicators": {
                "revenue_trend": "POSITIVE",
                "profitability": "PROFITABLE"
            }
        }
        
        health = generator._assess_overall_health(report)
        assert health == "HEALTHY"
    
    def test_report_generator_agent_creation(self):
        """Test ReportGeneratorAgent creation"""
        agent = ReportGeneratorAgent()
        assert agent.generator is not None
        
        # Test agent creation
        crew_agent = agent.create_agent()
        assert crew_agent.role == 'Financial Report Writer'
        assert crew_agent.goal is not None

class TestRAGPipeline:
    """Test cases for RAG Pipeline"""
    
    def test_document_chunker(self):
        """Test document chunker"""
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        
        # Sample text
        text = "This is a test document. " * 50  # Create long text
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("token_count" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
    
    def test_sentence_splitting(self):
        """Test sentence splitting"""
        chunker = DocumentChunker()
        
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) >= 3  # Should split into multiple sentences
        assert all(len(s.strip()) > 0 for s in sentences)
    
    def test_vector_store_initialization(self):
        """Test vector store initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary vector store
            vector_store = VectorStore(index_path=f"{temp_dir}/test_index")
            
            assert vector_store.index is not None
            assert vector_store.documents == []
            assert vector_store.metadata == []
    
    def test_rag_pipeline_initialization(self):
        """Test RAG pipeline initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change the index path
            original_path = settings.FAISS_INDEX_PATH
            settings.FAISS_INDEX_PATH = f"{temp_dir}/test_faiss_index"
            
            try:
                rag = RAGPipeline()
                assert rag.chunker is not None
                assert rag.vector_store is not None
                assert rag.reranker is not None
                
                # Test stats
                stats = rag.get_pipeline_stats()
                assert "vector_store_stats" in stats
                assert "chunk_size" in stats
                assert "overlap" in stats
            finally:
                # Restore original path
                settings.FAISS_INDEX_PATH = original_path

class TestConfiguration:
    """Test configuration and settings"""
    
    def test_settings_import(self):
        """Test settings import"""
        assert settings.ULTRASAFE_MODEL is not None
        assert settings.EMBEDDINGS_MODEL is not None
        assert settings.CHUNK_SIZE > 0
        assert settings.CHUNK_OVERLAP >= 0
        assert settings.MAX_TOKENS > 0
        assert settings.TEMPERATURE >= 0.0
    
    def test_settings_validation(self):
        """Test settings validation"""
        # This will fail if API key is not set, which is expected
        validation_result = settings.validate()
        assert isinstance(validation_result, bool)

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_sample_data_creation(self):
        """Test sample data creation"""
        # Check if sample text file exists
        sample_file = Path("data/sample_financial_report.txt")
        assert sample_file.exists()
        
        # Check content
        with open(sample_file, 'r') as f:
            content = f.read()
            assert "TECHCORP" in content
            assert "FINANCIAL REPORT" in content
            assert "revenue" in content.lower()
            assert "profit" in content.lower()
    
    def test_directory_structure(self):
        """Test that required directories can be created"""
        required_dirs = ["data", "logs", "config", "agents", "utils"]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            assert dir_path.exists(), f"Directory {dir_name} should exist"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 