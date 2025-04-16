import streamlit as st
import os
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ Google Gemini API Key not found. Set 'GEMINI_API_KEY' in .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

st.set_page_config(page_title="LLM Document Q&A", layout="wide")

# UI Title
st.markdown("""
    <h1 style='text-align: center;'>📄💬 Document Chat with Gemini LLM</h1>
    <hr>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# Function to create FAISS index
def create_faiss_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store

# Function to retrieve the most relevant chunks
def retrieve_relevant_text(query, vector_store):
    docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Function to get an answer from Gemini with streaming
def ask_gemini(query, context):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response_stream = model.generate_content(f"Answer based on the provided document:\n\n{context}\n\nQuestion: {query}\nAnswer:")
    
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text 

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# File uploader
uploaded_files = st.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.success("✅ Files uploaded successfully!")
    
    if "processed_files" not in st.session_state or st.session_state.processed_files != uploaded_files:
        with st.spinner("🔍 Extracting text and processing document..."):
            combined_text = ""
            for file in uploaded_files:
                combined_text += extract_text_from_pdf(file) + "\n"
            
            st.session_state.vector_store = create_faiss_index(combined_text)
            st.session_state.processed_files = uploaded_files
        
        st.success("✅ Documents processed successfully! You can now ask questions.")

# Chat UI
st.markdown("## 🗨️ Chat")
chat_container = st.container()

for entry in st.session_state.chat_history:
    with chat_container.chat_message(entry["role"]):
        st.markdown(entry["content"])

# Question input
if st.session_state.vector_store:
    query = st.chat_input("Ask a question about the document...")
    if query:
        with chat_container.chat_message("user"):
            st.markdown(query)

        with st.spinner("🤖 Retrieving answer..."):
            context = retrieve_relevant_text(query, st.session_state.vector_store)
            response_generator = ask_gemini(query, context)

            # Use st.empty() to update UI dynamically
            assistant_message = st.empty()
            answer_text = ""

            for chunk in response_generator:
                answer_text += chunk  # Append new content
                assistant_message.markdown(answer_text)  # Update UI dynamically

        # Append to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})