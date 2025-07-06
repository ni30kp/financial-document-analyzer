#!/usr/bin/env python3
"""
System Diagnostics Script for Financial Agent System

This script checks for common issues and provides fixes:
- FAISS GPU warnings
- OpenTelemetry conflicts
- Index management
- Dependency issues
- Performance optimization
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_basic_logging():
    """Setup basic logging for diagnostics"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check Python version compatibility"""
    print(f"🐍 Python Version: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.8 or higher")
        return False

def check_virtual_environment():
    """Check if virtual environment is activated"""
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        print(f"✅ Virtual environment active: {venv_path}")
        return True
    else:
        print("⚠️  No virtual environment detected")
        print("💡 Activate with: source financial_agent_env/bin/activate")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'crewai', 'langchain', 'openai', 'faiss', 'streamlit', 
        'plotly', 'pydantic', 'sentence_transformers', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_system_imports():
    """Check if system components can be imported"""
    print("\n🔧 Checking System Imports...")
    
    try:
        from config.settings import settings
        print("✅ Settings imported successfully")
        
        from financial_agent_system import FinancialAgentSystem
        print("✅ FinancialAgentSystem imported successfully")
        
        from utils.rag_pipeline import RAGPipeline
        print("✅ RAGPipeline imported successfully")
        
        from utils.index_manager import IndexManager
        print("✅ IndexManager imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

def check_faiss_configuration():
    """Check FAISS configuration and GPU warnings"""
    print("\n🔍 Checking FAISS Configuration...")
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
        
        # Check if GPU is available but we're using CPU
        if hasattr(faiss, 'get_num_gpus'):
            gpu_count = faiss.get_num_gpus()
            print(f"📊 Available GPUs: {gpu_count}")
            
            if gpu_count > 0:
                print("💡 GPU available but using CPU-only FAISS (recommended)")
            else:
                print("✅ CPU-only FAISS configuration")
        
        # Test index creation
        test_index = faiss.IndexFlatIP(128)
        print("✅ FAISS index creation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS error: {e}")
        return False

def check_opentelemetry_setup():
    """Check OpenTelemetry configuration"""
    print("\n📡 Checking OpenTelemetry Setup...")
    
    otel_disabled = os.environ.get("OTEL_SDK_DISABLED", "false").lower() == "true"
    otel_instrumentations = os.environ.get("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "")
    
    if otel_disabled:
        print("✅ OpenTelemetry SDK disabled (prevents conflicts)")
    else:
        print("⚠️  OpenTelemetry SDK not disabled")
        print("💡 Set OTEL_SDK_DISABLED=true to prevent conflicts")
    
    if otel_instrumentations == "all":
        print("✅ OpenTelemetry instrumentations disabled")
    else:
        print("⚠️  OpenTelemetry instrumentations not disabled")
        print("💡 Set OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=all")
    
    return otel_disabled and otel_instrumentations == "all"

def check_index_status():
    """Check index status and health"""
    print("\n📊 Checking Index Status...")
    
    try:
        from utils.index_manager import IndexManager
        
        manager = IndexManager()
        info = manager.get_index_info()
        
        print(f"Index Path: {info['index_path']}")
        print(f"Index Exists: {info['index_exists']}")
        print(f"Document Count: {info['document_count']}")
        print(f"Total Size: {info['total_size_mb']:.2f} MB")
        
        if info['index_exists']:
            print("✅ Index is healthy")
            
            # Check if index is too large
            if info['total_size_mb'] > 100:
                print("⚠️  Index is quite large (>100MB)")
                print("💡 Consider optimizing or clearing old documents")
            
            # Check document count
            if info['document_count'] > 1000:
                print("⚠️  High document count (>1000)")
                print("💡 Consider index optimization")
        else:
            print("📝 No existing index found (will be created on first use)")
        
        return True
        
    except Exception as e:
        print(f"❌ Index check error: {e}")
        return False

def check_api_configuration():
    """Check API configuration"""
    print("\n🔑 Checking API Configuration...")
    
    try:
        from config.settings import settings
        
        if settings.ULTRASAFE_API_KEY == "your_ultrasafe_api_key_here":
            print("❌ UltraSafe API key not configured")
            print("💡 Update your .env file with the correct API key")
            return False
        else:
            print("✅ UltraSafe API key configured")
            
        print(f"📡 API Base URL: {settings.ULTRASAFE_BASE_URL}")
        print(f"🤖 Model: {settings.ULTRASAFE_MODEL}")
        
        # Check for URL duplication issue
        if "/chat/completions" in settings.ULTRASAFE_BASE_URL:
            print("⚠️  Base URL contains '/chat/completions' - this may cause duplication")
            print("💡 Remove '/chat/completions' from base URL")
            return False
        else:
            print("✅ Base URL format is correct")
        
        return True
        
    except Exception as e:
        print(f"❌ API configuration error: {e}")
        return False

def run_system_test():
    """Run a quick system test"""
    print("\n🧪 Running System Test...")
    
    try:
        from financial_agent_system import FinancialAgentSystem
        
        # Initialize system
        system = FinancialAgentSystem()
        
        # Get system status
        status = system.get_system_status()
        
        print("✅ System initialized successfully")
        print(f"📊 System Stats: {status['system_stats']}")
        print(f"🔍 Settings Valid: {status['settings_valid']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

def provide_recommendations():
    """Provide optimization recommendations"""
    print("\n💡 Recommendations:")
    print("1. Always use virtual environment")
    print("2. Keep dependencies updated")
    print("3. Monitor index size and optimize regularly")
    print("4. Use CPU-only FAISS for stability")
    print("5. Disable OpenTelemetry to prevent conflicts")
    print("6. Configure proper logging levels")
    print("7. Backup index before major operations")

def main():
    """Main diagnostics function"""
    print("🔍 Financial Agent System Diagnostics")
    print("=" * 40)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger = setup_basic_logging()
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("System Imports", check_system_imports),
        ("FAISS Configuration", check_faiss_configuration),
        ("OpenTelemetry Setup", check_opentelemetry_setup),
        ("Index Status", check_index_status),
        ("API Configuration", check_api_configuration),
        ("System Test", run_system_test)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"🔍 {check_name}")
        print("-" * 50)
        
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 DIAGNOSTICS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name:.<30} {status}")
    
    print(f"\n📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! System is healthy.")
    else:
        print("⚠️  Some issues detected. Please review the recommendations.")
    
    provide_recommendations()

if __name__ == "__main__":
    main() 