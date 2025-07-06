import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px

from financial_agent_system import FinancialAgentSystem
from utils.multihop_rag import MultiHopRAGPipeline
from config.settings import Settings

st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📊",
    layout="wide"
)

# Session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'multihop' not in st.session_state:
    st.session_state.multihop = None
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'chat' not in st.session_state:
    st.session_state.chat = []

def init_system():
    try:
        if st.session_state.system is None:
            st.session_state.system = FinancialAgentSystem()
        if st.session_state.multihop is None:
            st.session_state.multihop = MultiHopRAGPipeline(st.session_state.system.rag_pipeline)
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def process_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        with st.spinner("Processing..."):
            result = st.session_state.system.process_document(tmp_path)
            st.session_state.docs.append({
                'name': uploaded_file.name,
                'timestamp': datetime.now(),
                'result': result
            })
        
        os.unlink(tmp_path)
        return result
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def show_metrics(analysis):
    if 'financial_analysis' not in analysis:
        return
    
    metrics = analysis['financial_analysis']['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue = metrics.get('revenue', {})
        st.metric("Revenue", revenue.get('current_year', 'N/A'))
    
    with col2:
        profit = metrics.get('profit_metrics', {})
        st.metric("Net Income", profit.get('net_income', 'N/A'))
    
    with col3:
        ratios = metrics.get('key_ratios', {})
        st.metric("ROE", ratios.get('return_on_equity', 'N/A'))
    
    with col4:
        st.metric("EPS", ratios.get('earnings_per_share', 'N/A'))

def simple_chat():
    st.subheader("Simple Chat")
    
    if not st.session_state.docs:
        st.info("Upload a document first")
        return
    
    question = st.text_input("Ask a question:")
    
    if question and st.button("Ask"):
        try:
            context = st.session_state.system.rag_pipeline.retrieve_context(question)
            
            from openai import OpenAI
            settings = Settings()
            client = OpenAI(
                api_key=settings.ULTRASAFE_API_KEY,
                base_url=settings.ULTRASAFE_BASE_URL
            )
            
            response = client.chat.completions.create(
                model=settings.ULTRASAFE_MODEL,
                messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}],
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")
            
            st.session_state.chat.append({
                'q': question,
                'a': answer,
                'time': datetime.now()
            })
            
        except Exception as e:
            st.error(f"Error: {e}")

def multihop_chat():
    st.subheader("Multi-Hop Analysis")
    
    if not st.session_state.docs:
        st.info("Upload a document first")
        return
    
    question = st.text_input("Ask a complex question:")
    
    if question and st.button("Analyze"):
        try:
            with st.spinner("Processing..."):
                result = st.session_state.multihop.process_query(question)
            
            st.write(f"**Question:** {result.original_query}")
            st.write(f"**Answer:** {result.final_answer}")
            st.write(f"**Hops:** {result.total_hops}")
            st.write(f"**Confidence:** {result.confidence:.2f}")
            
            with st.expander("Reasoning Steps"):
                for hop in result.hops:
                    st.write(f"**Step {hop.hop_number}:** {hop.sub_query}")
                    st.write(f"Answer: {hop.reasoning}")
                    st.write("---")
            
        except Exception as e:
            st.error(f"Error: {e}")

def main():
    st.title("📊 Financial Document Analyzer")
    
    if not init_system():
        st.stop()
    
    tab = st.sidebar.radio("Select:", ["Upload", "Simple Chat", "Multi-Hop", "Results"])
    
    if tab == "Upload":
        st.header("Upload Document")
        
        uploaded_file = st.file_uploader("Choose PDF", type="pdf")
        
        if uploaded_file:
            if st.button("Process"):
                result = process_file(uploaded_file)
                if result:
                    st.success("Document processed!")
                    show_metrics(result)
    
    elif tab == "Simple Chat":
        simple_chat()
    
    elif tab == "Multi-Hop":
        multihop_chat()
    
    elif tab == "Results":
        st.header("Results")
        
        if st.session_state.docs:
            for doc in st.session_state.docs:
                with st.expander(f"{doc['name']} - {doc['timestamp'].strftime('%H:%M')}"):
                    show_metrics(doc['result'])
        else:
            st.info("No documents processed yet")

if __name__ == "__main__":
    main() 