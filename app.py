import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIG ---

st.set_page_config(
    page_title="Smart HR Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---

st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .answer-box {
            background-color: #f0f7ff;
            border-left: 4px solid #0066cc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            color: #000000;
            font-weight: 500;
        }
        .source-box {
            background-color: #f5f5f5;
            border-left: 4px solid #666;
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---

with st.sidebar:
    st.markdown("## 📚 HR Assistant")
    st.markdown("---")
    st.markdown("""
    ### How to Use:
    1. **Upload Documents** - Add HR policy PDFs or DOCX files
    2. **Process** - Documents are indexed automatically
    3. **Ask Questions** - Query your HR policies
    4. **Get Answers** - Get accurate responses with sources
    """)
    st.markdown("---")
    st.markdown("""
    ### Supported Formats:
    - 📄 PDF files
    - 📝 Word documents (DOCX)
    """)
    st.markdown("---")
    st.markdown("""
    ### ⚡ Features:
    - AI-powered search & retrieval
    - Source citations
    - Persistent storage
    """)

# --- MAIN CONTENT ---

st.title("📄 Smart HR Document Assistant")
st.markdown("*Your intelligent HR policy search engine powered by Google Gemini*")
st.markdown("---")

# --- LOAD ENV ---


def get_api_key():
    load_dotenv()

    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        elif "GEMINI" in st.secrets and "GEMINI_API_KEY" in st.secrets["GEMINI"]:
            api_key = st.secrets["GEMINI"]["GEMINI_API_KEY"]
    except Exception:
        api_key = None

    # Fall back to local .env for development.
    return api_key or os.getenv("GEMINI_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error(
        "Missing GEMINI_API_KEY. Add it to Streamlit secrets or .env and reload the app."
    )
    st.stop()

with st.sidebar:
    if st.button("Reset Vector DB"):
        shutil.rmtree("db", ignore_errors=True)
        load_vector_db.clear()
        st.success("✅ Vector database cleared. Re-upload files to rebuild it.")

# --- FILE UPLOAD SECTION ---

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload HR policy documents to search through"
    )

with col2:
    if uploaded_files:
        st.metric("Files Uploaded", len(uploaded_files))

st.markdown("---")

# --- LOAD DOCUMENTS ---

def load_documents(files):
    documents = []
    
    for file in files:
        file_path = f"/tmp/{file.name}"
        
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        if file.size > 10 * 1024 * 1024:
            st.warning(f"{file.name} is larger than 10MB and may be skipped.")
            continue

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredWordDocumentLoader(file_path)

        try:
            docs = loader.load()
        except Exception as e:
            st.warning(f"Skipping {file.name}: {e}")
            continue

        for doc in docs:
            doc.metadata["source"] = file.name
            doc.metadata["page"] = doc.metadata.get("page", "N/A")
        
        documents.extend(docs)
    
    return documents

@st.cache_resource
def get_embeddings(api_key):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

@st.cache_resource
def load_vector_db(api_key, persist_directory: str):
    embeddings = get_embeddings(api_key)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

# --- MAIN ---

if api_key and uploaded_files:
    db_path = "db"
    embeddings = get_embeddings(api_key)
    llm = get_llm(api_key)
    vector_db = None

    if os.path.exists(db_path):
        try:
            vector_db = load_vector_db(api_key, db_path)
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("✅ Using existing vector database")
                with col2:
                    st.caption("No re-indexing needed")
        except Exception:
            st.warning(
                "Existing vector database could not be loaded and will be rebuilt."
            )
            shutil.rmtree(db_path, ignore_errors=True)
            load_vector_db.clear()
            vector_db = None

    if vector_db is None:
        with st.spinner("🔄 Processing documents..."):
            docs = load_documents(uploaded_files)
            if not docs:
                st.warning("No valid documents were loaded.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )
            vector_db.persist()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(uploaded_files))
        with col2:
            st.metric("Pages/Chunks", len(chunks))
        with col3:
            st.metric("Status", "Ready")

        st.success("✅ Documents indexed successfully!")

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""You are a professional HR policy assistant designed to provide accurate, factual information from organizational documents.

INSTRUCTIONS:
1. Answer ONLY based on the provided context from HR documents
2. Be precise and cite specific policies when applicable
3. If information spans multiple sections, synthesize a coherent answer
4. Use professional language appropriate for HR communications
5. Do not provide personal interpretations or legal advice beyond what's stated
6. If the answer cannot be found in the provided documents, respond with: "Not found in provided documents."

CONTEXT FROM HR DOCUMENTS:
{context}

EMPLOYEE QUESTION:
{question}

RESPONSE:
""")
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "docs": retriever
        }
        | RunnableLambda(lambda x: {
            "answer": llm.invoke(prompt.invoke(x)).content,
            "docs": x["docs"]
        })
    )
    
    st.markdown("---")
    st.subheader("❓ Ask a Question")
    
    query = st.text_area(
        "Enter your question",
        placeholder="e.g., What is the sick leave policy? How much annual leave am I entitled to?",
        height=100,
        help="Ask any question about the HR policies in your documents"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    st.markdown("---")
    
    if search_button:
        if not query.strip():
            st.warning("⚠️ Please enter a question before searching!")
        else:
            with st.spinner("🤔 Finding relevant information..."):
                result = rag_chain.invoke(query)
            
            # Answer Section
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Sources Section
            st.markdown("### 📎 Sources")
            
            seen = set()
            source_count = 0
            for doc in result["docs"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                
                key = (source, page)
                if key not in seen:
                    source_count += 1
                    with st.expander(f"📄 {source} (Page {page})", expanded=(source_count == 1)):
                        st.caption(content_preview)
                    seen.add(key)

else:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if not api_key:
            st.error("🔐 Configuration Error")
            st.markdown("""
            **GEMINI_API_KEY not found**
            
            Please ensure `.env` file contains your Google Gemini API key:
            ```
            GEMINI_API_KEY=your-api-key-here
            ```
            """)
    
    with col2:
        if not uploaded_files:
            st.warning("📤 Ready to Start")
            st.markdown("""
            **Upload documents to begin**
            
            1. Click 'Browse files' above
            2. Select PDF or DOCX files
            3. Wait for processing
            4. Start asking questions!
            """)
 