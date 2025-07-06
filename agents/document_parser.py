"""
Document Parser Agent for extracting text from PDF financial documents
"""
import fitz  # PyMuPDF
import pdfplumber
import re
import logging
from pathlib import Path
from crewai import Agent, Task
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf']
        
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF error: {e}")
            raise
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"pdfplumber error: {e}")
            raise
    
    def extract_tables_pdfplumber(self, pdf_path: str) -> list:
        try:
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            return tables
        except Exception as e:
            logger.error(f"Table extraction error: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*of\s*\d+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\(\)\-\+\=\.\,\;\:]', '', text)
        
        # Normalize financial figures
        text = re.sub(r'\$\s+', '$', text)
        text = re.sub(r'\%\s+', '%', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def extract_and_clean(self, pdf_path: str) -> dict:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Unsupported file format: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Try PyMuPDF first
            raw_text = self.extract_text_pymupdf(pdf_path)
            extraction_method = "PyMuPDF"
            
            # If PyMuPDF fails or returns empty, try pdfplumber
            if not raw_text.strip():
                raw_text = self.extract_text_pdfplumber(pdf_path)
                extraction_method = "pdfplumber"
                
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying pdfplumber: {e}")
            raw_text = self.extract_text_pdfplumber(pdf_path)
            extraction_method = "pdfplumber"
        
        # Extract tables
        tables = self.extract_tables_pdfplumber(pdf_path)
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        result = {
            "file_path": pdf_path,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "tables": tables,
            "extraction_method": extraction_method,
            "word_count": len(cleaned_text.split()),
            "character_count": len(cleaned_text)
        }
        
        logger.info(f"Extracted {result['word_count']} words using {extraction_method}")
        return result

class DocumentParserAgent:
    def __init__(self):
        self.parser = DocumentParser()
        
    def create_agent(self) -> Agent:
        return Agent(
            role='Document Parser',
            goal='Extract and clean text from PDF financial documents',
            backstory="""You are a document processing specialist with experience 
            in extracting information from financial PDFs.""",
            verbose=True,
            allow_delegation=False
        )
    
    def create_task(self, pdf_path: str) -> Task:
        return Task(
            description=f"Extract text from PDF: {pdf_path}",
            agent=self.create_agent(),
            expected_output="Structured document data with text, tables, and metadata"
        )
    
    def execute(self, pdf_path: str) -> dict:
        try:
            return self.parser.extract_and_clean(pdf_path)
        except Exception as e:
            logger.error(f"Error in document parsing: {e}")
            raise 