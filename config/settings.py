"""
Configuration settings
"""
import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Configure OpenTelemetry to avoid conflicts
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="opentelemetry")

class Settings:
    """Configuration settings"""
    
    # UltraSafe API Configuration
    ULTRASAFE_API_KEY: str = os.getenv("ULTRASAFE_API_KEY", "your_ultrasafe_api_key_here")
    ULTRASAFE_BASE_URL: str = os.getenv("ULTRASAFE_BASE_URL", "https://api.us.inc/usf/v1/hiring")
    ULTRASAFE_MODEL: str = os.getenv("ULTRASAFE_MODEL", "usf1-mini")
    
    # Vector Database Configuration
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/financial_agent.log")
    
    # System Configuration
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Performance Configuration
    FAISS_CPU_ONLY: bool = os.getenv("FAISS_CPU_ONLY", "true").lower() == "true"
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", "3"))
    
    # Paths
    DATA_DIR: str = "./data"
    LOGS_DIR: str = "./logs"
    
    def validate(self) -> bool:
        """Check if API key is set"""
        if self.ULTRASAFE_API_KEY == "your_ultrasafe_api_key_here":
            return False
        return True
    
    def setup_logging(self) -> None:
        """Setup logging"""
        # Create logs directory
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        
        # Quiet down noisy loggers
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("faiss").setLevel(logging.WARNING)
        logging.getLogger("opentelemetry").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

# Global settings instance
settings = Settings() 