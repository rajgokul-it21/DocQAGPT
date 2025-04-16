import streamlit as st
import os
import pdfplumber
import pickle
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
st.title("📄💬 Document Chat with Gemini LLM")

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

    with open("vector_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store

# Function to retrieve the most relevant chunks
def retrieve_relevant_text(query, vector_store):
    docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Function to get an answer from Gemini
def ask_gemini(query, context):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(f"Answer based on the provided document:\n\n{context}\n\nQuestion: {query}\nAnswer:")
    return response.text if response else "No response received."

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Extract text
    with st.spinner("🔍 Extracting text from the document..."):
        doc_text = extract_text_from_pdf(uploaded_file)

    # Process and store FAISS index
    if st.button("⚡ Process Document"):
        with st.spinner("📚 Creating document embeddings..."):
            vector_store = create_faiss_index(doc_text)
        st.success("✅ Document processed successfully!")

# Load FAISS index if it exists
vector_store = None
if os.path.exists("vector_store.pkl"):
    with open("vector_store.pkl", "rb") as f:
        vector_store = pickle.load(f)

# Question input
if vector_store:
    query = st.text_input("💬 Ask a question about the document:")
    if query:
        with st.spinner("🤖 Retrieving answer..."):
            context = retrieve_relevant_text(query, vector_store)
            answer = ask_gemini(query, context)
        st.write("### 🤖 Gemini LLM Answer:")
        st.info(answer)

