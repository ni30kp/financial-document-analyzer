# Financial Document Analyzer 🏦

A production-ready multi-agent system for processing financial PDF reports and extracting key metrics using **CrewAI orchestration**, **RAG pipeline**, and **UltraSafe API integration**.

## ✨ Features

### 🤖 Multi-Agent Architecture
- **DocumentParserAgent**: Intelligent PDF text and table extraction
- **FinancialAnalysisAgent**: Advanced financial metrics analysis and trend identification  
- **ReportGeneratorAgent**: Executive summary and insights generation

### 🔍 Advanced RAG Pipeline
- **FAISS Vector Database**: High-performance similarity search
- **Multi-Hop RAG**: Complex query processing with reasoning chains
- **Intelligent Chunking**: Optimized document segmentation
- **Semantic Search**: Context-aware document retrieval

### 🌐 User Interfaces
- **Streamlit Web App**: Modern UI with document upload and interactive chat
- **CLI Interface**: Command-line tool for batch processing
- **REST API Ready**: Easily deployable as a web service

### 🛡️ Production Features
- **UltraSafe API Integration**: Enterprise-grade LLM service
- **Comprehensive Error Handling**: Robust error management
- **Performance Optimizations**: CPU-optimized FAISS, model caching
- **Health Diagnostics**: Built-in system health monitoring
- **Logging & Monitoring**: Comprehensive logging system

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (tested with Python 3.9.6 and 3.13.3)
- **UltraSafe API key**
- **macOS/Linux** (Windows support via WSL)

### One-Line Installation

```bash
git clone https://github.com/your-username/financial-document-analyzer.git
cd financial-document-analyzer
chmod +x setup_macos.sh && ./setup_macos.sh
```

The setup script will:
- ✅ Create and configure virtual environment
- ✅ Install all dependencies with compatible versions
- ✅ Set up FAISS CPU-only configuration
- ✅ Configure OpenTelemetry and logging
- ✅ Create sample configuration files
- ✅ Run comprehensive system diagnostics

### Manual Configuration

After installation, configure your API credentials:

```bash
# Copy the environment template
cp env.template .env

# Edit with your API key
nano .env
```

**Required Environment Variables:**
```bash
# UltraSafe API Configuration
ULTRASAFE_API_KEY=your_api_key_here
ULTRASAFE_BASE_URL=https://api.us.inc/usf/v1/hiring
ULTRASAFE_MODEL=usf1-mini

# System Configuration
FAISS_ENABLE_GPU=0
CUDA_VISIBLE_DEVICES=""
OTEL_SDK_DISABLED=true
OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=all
```

## 📖 Usage

### 🖥️ Command Line Interface

```bash
# Activate virtual environment
source financial_agent_env/bin/activate

# Run system diagnostics
python diagnose_system.py

# Demo mode (uses sample data)
python main.py --demo

# Process single document
python main.py --file path/to/financial_report.pdf

# Batch processing
python main.py --batch file1.pdf file2.pdf file3.pdf

# Interactive mode
python main.py --interactive
```

### 🌐 Web Interface

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

**Features:**
- 📁 **Document Upload**: Drag-and-drop PDF processing
- 💬 **Interactive Chat**: Ask questions about your documents
- 🔍 **Multi-Hop Analysis**: Complex analytical queries
- 📊 **Results Dashboard**: Visual metrics and insights
- 📈 **Performance Metrics**: System performance monitoring

### 🔧 System Health Check

```bash
# Run comprehensive diagnostics
python diagnose_system.py
```

**Diagnostic Checks:**
- ✅ Python version compatibility
- ✅ Virtual environment validation
- ✅ Dependencies installation
- ✅ System imports verification
- ✅ FAISS configuration
- ✅ OpenTelemetry setup
- ✅ Vector index health
- ✅ API configuration
- ✅ End-to-end system test

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Financial Document Analyzer              │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (Streamlit)    │    CLI Interface (main.py)  │
├─────────────────────────────────────────────────────────────┤
│                 FinancialAgentSystem                        │
├─────────────────────────────────────────────────────────────┤
│  DocumentParserAgent │ FinancialAnalysisAgent │ ReportAgent │
├─────────────────────────────────────────────────────────────┤
│                    RAG Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│  Document Chunker │ Vector Store (FAISS) │ Multi-Hop RAG   │
├─────────────────────────────────────────────────────────────┤
│              UltraSafe API │ SentenceTransformers           │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
financial-document-analyzer/
├── 🏗️ Core System
│   ├── financial_agent_system.py    # Main orchestrator
│   ├── main.py                      # CLI entry point
│   └── streamlit_app.py             # Web interface
├── 🤖 Agents
│   ├── agents/
│   │   ├── document_parser.py       # DocumentParserAgent
│   │   ├── analysis_agent.py        # FinancialAnalysisAgent
│   │   └── report_generator.py      # ReportGeneratorAgent
├── 🔧 Utilities
│   ├── utils/
│   │   ├── rag_pipeline.py         # RAG implementation
│   │   ├── multihop_rag.py         # Multi-hop RAG
│   │   └── index_manager.py        # Index management
├── ⚙️ Configuration
│   ├── config/
│   │   └── settings.py             # Settings management
├── 🧪 Testing & Diagnostics
│   ├── tests/                      # Test suite
│   ├── diagnose_system.py          # System diagnostics
│   └── setup_macos.sh             # Installation script
├── 📊 Data
│   ├── data/
│   │   ├── faiss_index/           # Vector database
│   │   ├── results/               # Processing results
│   │   └── sample_financial_report.pdf
└── 📋 Documentation
    ├── README.md                   # This file
    ├── requirements.txt            # Dependencies
    └── env.template               # Configuration template
```

## 🔧 API Reference

### FinancialAgentSystem

```python
from financial_agent_system import FinancialAgentSystem

# Initialize system
system = FinancialAgentSystem()

# Process single document
result = system.process_document(
    file_path="report.pdf",
    use_rag=True,
    analysis_type="comprehensive"
)

# Batch processing
results = system.batch_process([
    "q1_report.pdf", 
    "q2_report.pdf"
])

# Get system statistics
stats = system.get_stats()
print(f"Documents indexed: {stats['total_documents']}")
print(f"Index size: {stats['index_size_mb']} MB")
```

### RAG Pipeline

```python
from utils.rag_pipeline import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline()

# Add document to index
rag.index_document({
    "file_path": "document.pdf",
    "content": "document_text",
    "metadata": {"company": "ACME Corp", "year": 2023}
})

# Retrieve relevant context
context = rag.retrieve_context(
    query="What was the revenue growth?",
    top_k=5,
    rerank=True
)

# Get pipeline statistics
stats = rag.get_pipeline_stats()
```

### Multi-Hop RAG

```python
from utils.multihop_rag import MultiHopRAGPipeline

# Initialize with existing RAG pipeline
multihop = MultiHopRAGPipeline(rag_pipeline)

# Process complex query
result = multihop.process_query(
    "Compare Q1 vs Q4 revenue and explain the key factors"
)

print(f"Answer: {result.final_answer}")
print(f"Reasoning: {result.reasoning_chain}")
print(f"Confidence: {result.confidence}")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ULTRASAFE_API_KEY` | UltraSafe API key | - | ✅ |
| `ULTRASAFE_BASE_URL` | API base URL | `https://api.us.inc/usf/v1/hiring` | ✅ |
| `ULTRASAFE_MODEL` | Model name | `usf1-mini` | ✅ |
| `FAISS_INDEX_PATH` | Vector database path | `./data/faiss_index` | ❌ |
| `EMBEDDINGS_MODEL` | Embeddings model | `all-MiniLM-L6-v2` | ❌ |
| `MAX_TOKENS` | Max tokens per API call | `4000` | ❌ |
| `TEMPERATURE` | LLM temperature | `0.1` | ❌ |
| `CHUNK_SIZE` | Document chunk size | `1000` | ❌ |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` | ❌ |

### Performance Tuning

```python
# config/settings.py
PERFORMANCE_SETTINGS = {
    "faiss_enable_gpu": False,           # CPU-only for compatibility
    "model_cache_size": 3,               # Number of models to cache
    "max_concurrent_requests": 5,        # API rate limiting
    "chunk_processing_batch_size": 100,  # Batch processing size
    "vector_search_timeout": 30,         # Search timeout (seconds)
}
```

## 🧪 Testing

### Run Test Suite

```bash
# Activate virtual environment
source financial_agent_env/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_multihop_rag.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### System Diagnostics

```bash
# Full system health check
python diagnose_system.py

# Quick component test
python -c "
from financial_agent_system import FinancialAgentSystem
system = FinancialAgentSystem()
print('✅ System initialized successfully')
"
```

## 📊 Performance Metrics

**Typical Performance:**
- **Document Processing**: 2-5 seconds per PDF
- **Vector Search**: <100ms for 1000+ documents
- **Multi-Hop Analysis**: 10-30 seconds for complex queries
- **Memory Usage**: ~500MB for 1000 documents
- **Storage**: ~1MB per 1000 document chunks

## 🔍 Troubleshooting

### Common Issues

**1. Virtual Environment Issues**
```bash
# Ensure virtual environment is activated
source financial_agent_env/bin/activate

# Verify Python version
python --version  # Should be 3.9+

# Reinstall if needed
rm -rf financial_agent_env
./setup_macos.sh
```

**2. API Configuration Issues**
```bash
# Check API configuration
python -c "
from config.settings import settings
print(f'API URL: {settings.ULTRASAFE_BASE_URL}')
print(f'API Key: {settings.ULTRASAFE_API_KEY[:10]}...')
"

# Test API connection
python diagnose_system.py
```

**3. FAISS/Vector Database Issues**
```bash
# Check FAISS configuration
python -c "
import faiss
print(f'FAISS version: {faiss.__version__}')
print('✅ FAISS working correctly')
"

# Reset vector database
rm -rf data/faiss_index*
python main.py --demo  # Will rebuild index
```

**4. Dependencies Issues**
```bash
# Check installed packages
pip list | grep -E "crewai|langchain|faiss|streamlit"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with debug output
python main.py --demo --verbose

# Check system logs
tail -f logs/system.log
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-username/financial-document-analyzer.git
cd financial-document-analyzer

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI** for the multi-agent framework
- **LangChain** for the RAG pipeline foundation
- **FAISS** for efficient vector similarity search
- **Streamlit** for the web interface
- **UltraSafe** for the LLM API service

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-username/financial-document-analyzer/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/your-username/financial-document-analyzer/discussions)
- 📧 **Email**: your-email@example.com

---

**Made with ❤️ for financial document analysis** 