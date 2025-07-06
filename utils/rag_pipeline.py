"""
RAG Pipeline with FAISS Vector Database for Financial Document Retrieval
"""
import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

# Configure FAISS to use CPU only to avoid GPU warnings
os.environ["FAISS_ENABLE_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from config.settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Disable FAISS GPU warnings
faiss.omp_set_num_threads(1)

class ModelManager:
    """
    Singleton pattern for managing SentenceTransformer models to avoid multiple loading
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str) -> SentenceTransformer:
        """Get or create a SentenceTransformer model"""
        if model_name not in self._models:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]
    
    def clear_models(self):
        """Clear all loaded models"""
        self._models.clear()

class DocumentChunker:
    """
    Document chunker for splitting large documents into manageable chunks
    """
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_size + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk_data = {
                    "text": chunk_text,
                    "token_count": current_size,
                    "chunk_id": len(chunks),
                    "metadata": metadata or {}
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, self.overlap)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_data = {
                "text": chunk_text,
                "token_count": current_size,
                "chunk_id": len(chunks),
                "metadata": metadata or {}
            }
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristics
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Handle special cases for financial documents
        processed_sentences = []
        for sentence in sentences:
            # Don't split on decimal points in numbers
            if len(sentence) > 10 and not sentence.replace(',', '').replace('$', '').replace('%', '').replace('.', '').replace('-', '').isdigit():
                processed_sentences.append(sentence)
            else:
                # Join with previous sentence if too short
                if processed_sentences and len(sentence) < 20:
                    processed_sentences[-1] += '. ' + sentence
                else:
                    processed_sentences.append(sentence)
        
        return processed_sentences
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """
        Get sentences for overlap based on token count
        
        Args:
            sentences: List of sentences
            overlap_tokens: Number of tokens to overlap
            
        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        token_count = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break
        
        return overlap_sentences

class VectorStore:
    """
    FAISS-based vector store for document embeddings
    """
    
    def __init__(self, model_name: str = None, index_path: str = None):
        self.model_name = model_name or settings.EMBEDDINGS_MODEL
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        
        # Initialize embeddings model using singleton
        self.model_manager = ModelManager()
        self.embeddings_model = self.model_manager.get_model(self.model_name)
        self.embedding_dim = self.embeddings_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            index_file = f"{self.index_path}.index"
            metadata_file = f"{self.index_path}.metadata"
            documents_file = f"{self.index_path}.documents"
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                self.index = faiss.read_index(index_file)
                
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                self._create_new_index()
                
        except Exception as e:
            logger.warning(f"Failed to load index: {e}, creating new one")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Explicitly create CPU-only index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        # Ensure CPU-only execution
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            logger.info("GPU available but using CPU-only FAISS index")
        self.documents = []
        self.metadata = []
        logger.info("Created new CPU-only FAISS index")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            return
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embeddings_model.encode(texts, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend([chunk.get("metadata", {}) for chunk in chunks])
        
        logger.info(f"Added {len(texts)} chunks to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    "text": self.documents[idx],
                    "score": float(score),
                    "rank": i + 1,
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                }
                results.append(result)
        
        return results
    
    def save_index(self) -> None:
        """Save the FAISS index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save metadata
            with open(f"{self.index_path}.metadata", 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save documents
            with open(f"{self.index_path}.documents", 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name
        }

class Reranker:
    """
    Simple reranker for improving retrieval quality
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDINGS_MODEL
        # Use singleton model manager
        self.model_manager = ModelManager()
        self.embeddings_model = self.model_manager.get_model(self.model_name)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder or similarity scoring
        
        Args:
            query: Original query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings for query and documents
        query_embedding = self.embeddings_model.encode([query])
        doc_embeddings = self.embeddings_model.encode(texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Create reranked results
        reranked_docs = []
        for i, (doc, similarity) in enumerate(zip(documents, similarities)):
            reranked_doc = doc.copy()
            reranked_doc["rerank_score"] = float(similarity)
            reranked_doc["original_rank"] = doc.get("rank", i + 1)
            reranked_docs.append(reranked_doc)
        
        # Sort by rerank score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top_k results
        return reranked_docs[:top_k]

class RAGPipeline:
    """
    Complete RAG pipeline for financial document retrieval
    """
    
    def __init__(self):
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        
    def index_document(self, document_data: Dict[str, Any]) -> None:
        """
        Index a document for retrieval
        
        Args:
            document_data: Document data from parser
        """
        text = document_data.get("cleaned_text", "")
        metadata = {
            "file_path": document_data.get("file_path", ""),
            "extraction_method": document_data.get("extraction_method", ""),
            "word_count": document_data.get("word_count", 0),
            "indexed_at": str(datetime.now())
        }
        
        # Create chunks
        chunks = self.chunker.chunk_text(text, metadata)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Save index
        self.vector_store.save_index()
        
        logger.info(f"Indexed document with {len(chunks)} chunks")
    
    def retrieve_context(self, query: str, top_k: int = 5, rerank: bool = True) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            rerank: Whether to rerank results
            
        Returns:
            Concatenated context text
        """
        # Search for similar documents
        results = self.vector_store.search(query, k=top_k * 2)  # Get more for reranking
        
        if not results:
            return ""
        
        # Rerank if requested
        if rerank:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        # Concatenate context
        context_parts = []
        for result in results:
            context_parts.append(f"[Score: {result.get('rerank_score', result.get('score', 0)):.3f}] {result['text']}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"Retrieved context from {len(results)} documents")
        
        return context
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "chunk_size": self.chunker.chunk_size,
            "overlap": self.chunker.overlap,
            "reranker_model": self.reranker.model_name
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Example document data
    sample_doc = {
        "cleaned_text": "Apple Inc. reported record quarterly revenue of $97.3 billion for Q1 2023, representing a 5% increase year-over-year. The company's net income was $30.0 billion, with a profit margin of 30.8%. iPhone sales contributed $65.8 billion to total revenue, while Services revenue reached $20.8 billion. The company's balance sheet remains strong with $165.0 billion in cash and cash equivalents.",
        "file_path": "sample_apple_q1_2023.pdf",
        "extraction_method": "PyMuPDF",
        "word_count": 85
    }
    
    # Index the document
    rag.index_document(sample_doc)
    
    # Test retrieval
    query = "What was Apple's revenue and profit performance?"
    context = rag.retrieve_context(query, top_k=3)
    
    print(f"Query: {query}")
    print(f"Context: {context}")
    print(f"Pipeline Stats: {rag.get_pipeline_stats()}") 