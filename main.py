import streamlit as st
import os
import requests
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time

# Page configuration
st.set_page_config(
    page_title="Scheme Research AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .feature-card {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #e1e5e9;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .sidebar .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e1e5e9;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Load API key from .config file
load_dotenv(".config")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = 0

# Set up Gemini embedding and LLM
@st.cache_resource
def initialize_models():
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=gemini_api_key
    )
    return embedding, llm

embedding, llm = initialize_models()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍 Scheme Research AI</h1>
    <p>Intelligent Document Analysis & Question Answering System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("### 📂 Document Input")
    
    # URL input section
    st.markdown("#### 🌐 Web URLs")
    urls_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/document.pdf\nhttps://example.com/page",
        height=120,
        help="Enter web URLs or direct PDF links, one per line"
    )
    
    # File upload section
    st.markdown("#### 📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload PDF documents for analysis"
    )
    
    # Processing button
    st.markdown("---")
    process_btn = st.button(
        "🚀 Process Documents",
        use_container_width=True,
        disabled=st.session_state.processing
    )
    
    # Processing status
    if st.session_state.processing:
        st.info("🔄 Processing documents...")
        progress_bar = st.progress(0)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Documents", st.session_state.documents_processed)
    with col2:
        db_status = "✅ Ready" if os.path.exists("faiss_index_gemini/index.faiss") else "❌ Not Ready"
        st.metric("🗄️ Database", db_status)

# Main content area
if process_btn and not st.session_state.processing:
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing:
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    docs = []
    total_sources = len(urls) + (1 if uploaded_file else 0)
    current_progress = 0
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔄 Processing Documents")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Load from URLs
    if urls:
        for i, url in enumerate(urls):
            status_text.text(f"Loading URL {i+1}/{len(urls)}: {url}")
            try:
                if url.lower().endswith(".pdf"):
                    # Handle PDF URL
                    response = requests.get(url)
                    if response.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(response.content)
                            tmp_file_path = tmp_file.name
                        loader = PyPDFLoader(tmp_file_path)
                        docs.extend(loader.load())
                        st.success(f"✅ Loaded PDF from: {url}")
                    else:
                        st.error(f"❌ Failed to load PDF from: {url}")
                else:
                    # Handle HTML URL
                    loader = UnstructuredURLLoader(urls=[url])
                    docs.extend(loader.load())
                    st.success(f"✅ Loaded web page from: {url}")
                
                current_progress += 1
                progress_bar.progress(current_progress / total_sources)
                
            except Exception as e:
                st.error(f"❌ Error loading {url}: {str(e)}")

    # Load from uploaded PDF
    if uploaded_file:
        status_text.text("Loading uploaded PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs.extend(loader.load())
        st.success("✅ Uploaded PDF loaded successfully.")
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)

    if not docs:
        st.error("❌ Please input a valid URL or upload a PDF file.")
        st.session_state.processing = False
        st.stop()

    # Summarize the content
    status_text.text("Generating AI summary...")
    full_text = "\n".join([doc.page_content for doc in docs])
    summary_prompt = (
        "Summarize the following content in simple and concise terms. Focus on key points, benefits, eligibility criteria, and important details:\n\n"
        + full_text[:30000]  # Limit for token safety
    )
    
    with st.spinner("🤖 AI is analyzing the content..."):
        summary_response = llm.invoke(summary_prompt)
        summary = summary_response.content if hasattr(summary_response, 'content') else summary_response
        st.session_state["summary"] = summary

    # Chunk and Embed
    status_text.text("Creating vector database...")
    with st.spinner("🔍 Building searchable index..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embedding)
        db.save_local("faiss_index_gemini")
        st.session_state["db_ready"] = True
        st.session_state.documents_processed = len(docs)

    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    st.balloons()
    st.success("🎉 Documents processed successfully! You can now ask questions.")
    
    st.session_state.processing = False
    time.sleep(2)
    st.rerun()

# Load and provide Q&A interface
if os.path.exists("faiss_index_gemini/index.faiss"):
    db = FAISS.load_local(
        "faiss_index_gemini",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5})
    )

    # Main Q&A interface
    st.markdown("## 💬 Ask Questions About Your Documents")
    
    # Show summary in an attractive expandable box
    if "summary" in st.session_state:
        with st.expander("📋 **View AI-Generated Summary**", expanded=False):
            st.markdown(f"""
            <div class="summary-box">
                <h4>📄 Document Summary</h4>
                <p>{st.session_state["summary"]}</p>
            </div>
            """, unsafe_allow_html=True)

    # Question input with better styling
    st.markdown("### 🤔 What would you like to know?")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input(
            "",
            placeholder="e.g., What are the eligibility criteria? What are the benefits? How to apply?",
            label_visibility="collapsed"
        )
    with col2:
        ask_btn = st.button("🔍 Ask", use_container_width=True)

    # Suggested questions
    st.markdown("#### 💡 Suggested Questions:")
    question_cols = st.columns(3)
    
    suggested_questions = [
        "What are the main benefits?",
        "Who is eligible for this scheme?",
        "How can I apply?",
        "What documents are required?",
        "What is the application process?",
        "Are there any deadlines?"
    ]
    
    for i, question in enumerate(suggested_questions):
        if question_cols[i % 3].button(question, key=f"suggest_{i}"):
            user_query = question
            st.rerun()

    # Process question and show answer
    if user_query and (ask_btn or user_query):
        with st.spinner("🤖 AI is thinking..."):
            answer = qa.run(user_query)
        
        st.markdown(f"""
        <div class="answer-box">
            <h4>🎯 Answer:</h4>
            <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback section
        st.markdown("---")
        feedback_cols = st.columns([1, 1, 3])
        with feedback_cols[0]:
            if st.button("👍 Helpful"):
                st.success("Thank you for your feedback!")
        with feedback_cols[1]:
            if st.button("👎 Not Helpful"):
                st.info("We'll work on improving our responses!")

else:
    # Welcome screen when no documents are processed
    st.markdown("## 🎯 Welcome to Scheme Research AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🌐 Web Analysis</h4>
            <p>Analyze documents from any web URL or direct PDF links</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📄 Document Upload</h4>
            <p>Upload and analyze your own PDF documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI-Powered Q&A</h4>
            <p>Get instant answers to your questions about schemes and documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 **Get Started:** Use the sidebar to input URLs or upload documents, then click 'Process Documents'")
    
    # Quick tips
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - **URLs**: Enter government scheme pages, policy documents, or direct PDF links
    - **Questions**: Ask about eligibility, benefits, application process, deadlines, etc.
    - **Multiple Sources**: Process multiple documents at once for comprehensive analysis
    - **AI Summary**: Get an automatic summary of all processed documents
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔍 Scheme Research AI - Powered by Google Gemini & LangChain</p>
</div>
""", unsafe_allow_html=True)