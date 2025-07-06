# Financial Document Analyzer 

A multi-agent system that processes financial PDF reports and extracts key metrics. Built with CrewAI, RAG pipeline, and UltraSafe API. Works pretty well for analyzing quarterly reports, annual statements, and similar financial documents.

## Features

The system uses three specialized agents that work together:
- **DocumentParserAgent** - Extracts text and tables from PDFs (handles most common formats)
- **FinancialAnalysisAgent** - Analyzes financial metrics and identifies trends
- **ReportGeneratorAgent** - Creates executive summaries and insights

It includes a RAG pipeline with FAISS for document search, plus multi-hop reasoning for complex queries. You can use it via a web interface (Streamlit) or command line.

Some nice touches:
- CPU-optimized FAISS configuration (no GPU needed)
- Intelligent document chunking 
- Built-in health diagnostics
- Comprehensive error handling
- Works well for batch processing multiple documents

## Quick Start

### What You Need
- **Python 3.9.6** or **Python 3.13.3** - I've tested these versions extensively. Python 3.10-3.12 will probably work but I haven't tested them thoroughly. If you're having weird import errors or dependency conflicts, try switching to one of these tested versions.
- **UltraSafe API key** - Get this from their service
- **macOS or Linux** - Windows users can use WSL

### Installation

```bash
git clone https://github.com/your-username/financial-document-analyzer.git
cd financial-document-analyzer
chmod +x setup_macos.sh && ./setup_macos.sh
```

The setup script handles everything - creates a virtual environment, installs dependencies, configures FAISS for CPU-only operation, and runs diagnostics to make sure everything works. Takes about 2-3 minutes on a decent connection.

### Configuration

You'll need to add your API key after installation:

```bash
# Copy the environment template
cp env.template .env

# Edit with your API key
nano .env
```

The main thing you need to change is your API key:
```bash
ULTRASAFE_API_KEY=your_actual_api_key_here
```

The other settings should work as-is, but feel free to adjust them. The template has comments explaining what each one does. If you're getting "403 Invalid API key" errors, double-check that you've set the API key correctly in the .env file.

## Usage

### Command Line

```bash
# Activate virtual environment
source financial_agent_env/bin/activate

# Run system diagnostics (good to do first)
python diagnose_system.py

# Demo mode - uses sample data
python main.py --demo

# Process a single document
python main.py --file path/to/financial_report.pdf

# Process multiple documents
python main.py --batch file1.pdf file2.pdf file3.pdf

# Interactive mode
python main.py --interactive
```

### Web Interface

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The web interface lets you upload PDFs, ask questions about them, and run complex multi-hop analyses. It's pretty intuitive - just drag and drop your files and start asking questions.

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

**What to Expect:**
- **Document Processing**: 2-5 seconds per PDF (depends on size and complexity)
- **Vector Search**: Pretty fast - under 100ms for 1000+ documents
- **Multi-Hop Analysis**: 10-30 seconds for complex queries (this involves multiple API calls)
- **Memory Usage**: Around 500MB for 1000 documents in memory
- **Storage**: About 1MB per 1000 document chunks on disk

## Troubleshooting

### Common Issues

**Virtual Environment Problems**
```bash
# Make sure it's activated
source financial_agent_env/bin/activate

# Check Python version
python --version  # Should be 3.9.6 or 3.13.3

# If things are really broken, nuke and restart
rm -rf financial_agent_env
./setup_macos.sh
```

**API Key Issues**
```bash
# Check if your API key is set
python -c "
from config.settings import settings
print(f'API URL: {settings.ULTRASAFE_BASE_URL}')
print(f'API Key: {settings.ULTRASAFE_API_KEY[:10]}...')
"

# Run diagnostics to test API connection
python diagnose_system.py
```

**FAISS/Vector Database Issues**
```bash
# Check FAISS is working
python -c "
import faiss
print(f'FAISS version: {faiss.__version__}')
print('FAISS is working')
"

# If the vector database is corrupted, reset it
rm -rf data/faiss_index*
python main.py --demo  # This will rebuild the index
```

**Dependency Issues**
```bash
# Check what's installed
pip list | grep -E "crewai|langchain|faiss|streamlit"

# Nuclear option - reinstall everything
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