"""
CLI for the Financial Agent System
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from financial_agent_system import FinancialAgentSystem
from utils.create_sample_pdf import create_sample_pdf
from config.settings import settings

def setup_logging():
    Path(settings.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_sample_data():
    logger = logging.getLogger(__name__)
    try:
        logger.info("Creating sample PDF...")
        pdf_path = create_sample_pdf()
        logger.info(f"Created: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error creating sample: {e}")
        return None

def process_single_document(pdf_path: str, use_rag: bool = True):
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting system...")
        system = FinancialAgentSystem()
        
        logger.info(f"Processing: {pdf_path}")
        results = system.process_document(pdf_path, use_rag)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/results/results_{timestamp}.json"
        system.save_results(results, output_path)
        
        print_summary(results)
        return results
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def process_batch_documents(pdf_paths: list, use_rag: bool = True):
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting batch processing...")
        system = FinancialAgentSystem()
        
        logger.info(f"Processing {len(pdf_paths)} documents...")
        results = system.batch_process(pdf_paths, use_rag)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/results/batch_results_{timestamp}.json"
        system.save_results(results, output_path)
        
        print_batch_summary(results)
        return results
        
    except Exception as e:
        logger.error(f"Batch error: {e}")
        return None

def print_summary(results: dict):
    if not results:
        print("No results")
        return
    
    print("\n" + "-"*40)
    print("RESULTS")
    print("-"*40)
    
    metadata = results.get("metadata", {})
    print(f"Document: {metadata.get('document', 'Unknown')}")
    print(f"Processed: {metadata.get('timestamp', 'Unknown')}")
    
    analysis = results.get("financial_analysis", {})
    metrics = analysis.get("metrics", {})
    
    print(f"\nFINANCIAL METRICS:")
    
    revenue = metrics.get("revenue", {})
    if revenue:
        print(f"  Revenue: {revenue.get('current_year', 'N/A')}")
        print(f"  Growth: {revenue.get('growth_rate', 'N/A')}")
    
    profit = metrics.get("profit_metrics", {})
    if profit:
        print(f"  Net Income: {profit.get('net_income', 'N/A')}")
        print(f"  Margin: {profit.get('profit_margin', 'N/A')}")
    
    company = metrics.get("company_info", {})
    if company:
        print(f"  Company: {company.get('company_name', 'N/A')}")
        print(f"  Period: {company.get('reporting_period', 'N/A')}")
    
    report = results.get("executive_report", {})
    if report:
        dashboard = report.get("dashboard_summary", {})
        print(f"\nSUMMARY:")
        print(f"  Health: {dashboard.get('overall_health', 'N/A')}")
        print(f"  Risks: {len(dashboard.get('top_risks', []))}")
        print(f"  Trends: {len(dashboard.get('key_trends', []))}")
    
    print("-"*40)

def print_batch_summary(results: dict):
    if not results:
        print("No results")
        return
    
    print("\n" + "-"*40)
    print("BATCH RESULTS")
    print("-"*40)
    
    processed = results.get("processed", [])
    failed = results.get("failed", [])
    
    print(f"Total: {len(processed) + len(failed)}")
    print(f"Success: {len(processed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFAILURES:")
        for failure in failed:
            print(f"  - {failure.get('path', 'Unknown')}: {failure.get('error', 'Unknown')}")
    
    print("-"*40)

def run_demo():
    logger = setup_logging()
    logger.info("Demo mode...")
    
    pdf_path = create_sample_data()
    if not pdf_path:
        print("Failed to create sample data")
        return
    
    print("Processing sample...")
    results = process_single_document(pdf_path, use_rag=True)
    
    if results:
        print("\nDemo done!")
    else:
        print("Demo failed")

def main():
    parser = argparse.ArgumentParser(description='Financial Agent System')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--file', type=str, help='Process PDF file')
    parser.add_argument('--batch', nargs='+', help='Process multiple PDFs')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if args.demo:
        run_demo()
    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        
        process_single_document(args.file, use_rag=not args.no_rag)
    elif args.batch:
        existing_files = [f for f in args.batch if os.path.exists(f)]
        if not existing_files:
            print("No valid files")
            return
        
        process_batch_documents(existing_files, use_rag=not args.no_rag)
    else:
        print("Use --demo, --file, or --batch")

if __name__ == "__main__":
    main() 