"""
Financial document processing system
"""
import os
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

# Initialize settings and logging first
from config.settings import settings
settings.setup_logging()

from crewai import Agent, Task, Crew, Process
from agents.document_parser import DocumentParserAgent
from agents.analysis_agent import FinancialAnalysisAgent
from agents.report_generator import ReportGeneratorAgent
from utils.rag_pipeline import RAGPipeline
from utils.index_manager import IndexManager

logger = logging.getLogger(__name__)

class FinancialAgentSystem:
    def __init__(self):
        logger.info("Starting up...")
        
        # Initialize index manager
        self.index_manager = IndexManager()
        
        # Check index status
        index_info = self.index_manager.get_index_info()
        if index_info["index_exists"]:
            logger.info(f"Found existing index with {index_info['document_count']} documents ({index_info['total_size_mb']:.2f} MB)")
        
        # Initialize agents
        self.document_parser = DocumentParserAgent()
        self.analysis_agent = FinancialAnalysisAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline()
        
        self.stats = {
            "documents_processed": 0,
            "analysis_performed": 0,
            "reports_generated": 0,
            "system_initialized": datetime.now().isoformat()
        }
        
        self._setup_directories()
        logger.info("System ready")
        
    def _setup_directories(self):
        dirs = [settings.DATA_DIR, settings.LOGS_DIR, os.path.dirname(settings.FAISS_INDEX_PATH)]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "system_stats": self.stats,
            "index_info": self.index_manager.get_index_info(),
            "rag_stats": self.rag_pipeline.get_pipeline_stats(),
            "settings_valid": settings.validate()
        }
    
    def clear_index(self, create_backup: bool = True) -> bool:
        """Clear the document index"""
        logger.info("Clearing index...")
        return self.index_manager.clear_index(create_backup)
    
    def process_document(self, pdf_path: str, use_rag: bool = True) -> Dict[str, Any]:
        logger.info(f"Processing: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Document not found: {pdf_path}")
        
        # Parse document
        parsed_data = self.document_parser.execute(pdf_path)
        self.stats["documents_processed"] += 1
        
        # Index for RAG
        if use_rag:
            self.rag_pipeline.index_document(parsed_data)
        
        # Financial analysis
        context = ""
        if use_rag:
            query = f"financial metrics revenue profit {parsed_data.get('file_path', '')}"
            context = self.rag_pipeline.retrieve_context(query, top_k=3)
        
        analysis = self.analysis_agent.execute(parsed_data, context if context else None)
        self.stats["analysis_performed"] += 1
        
        # Generate report
        report = self.report_generator.execute(analysis)
        self.stats["reports_generated"] += 1
        
        return {
            "metadata": {
                "document": pdf_path,
                "timestamp": datetime.now().isoformat(),
                "use_rag": use_rag
            },
            "parsed_document": parsed_data,
            "financial_analysis": analysis,
            "executive_report": report,
            "stats": self.stats.copy()
        }
    
    def batch_process(self, pdf_paths: List[str], use_rag: bool = True) -> Dict[str, Any]:
        results = {"processed": [], "failed": []}
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_document(pdf_path, use_rag)
                results["processed"].append(result)
            except Exception as e:
                logger.error(f"Failed: {pdf_path}: {e}")
                results["failed"].append({"path": pdf_path, "error": str(e)})
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/results/results_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved to: {output_path}")
        return output_path
    
    def get_stats(self):
        return {
            **self.stats,
            "rag_stats": self.rag_pipeline.get_pipeline_stats(),
            "index_info": self.index_manager.get_index_info()
        } 