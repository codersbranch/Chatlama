
import streamlit as st
import requests
import time
from docx import Document
import ollama
import psutil
import platform
import tempfile
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to extract text from uploaded Word doc
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Load and process Word document
def load_docx_documents(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()

# Split text into chunks
def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)

# Create vector store
def create_vector_store(split_docs):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in split_docs]
    embeddings = embedder.encode(document_texts)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index, document_texts, embedder

# Retrieve relevant context
def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

# Function to get local Ollama models
def get_local_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            model_names = [model["name"] for model in models_data.get("models", [])]
            return model_names
        else:
            return [f"Error: {response.text}"]
    except Exception as e:
        return [f"Exception: {str(e)}"]    

# Function to chat with Ollama model
def chat_with_model(model, prompt):
    start_time = time.time()
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': 0.3, 'max_tokens': 200}
        )
        end_time = time.time()
        output = response['response']
        return output, round(end_time - start_time, 2)
    except Exception as e:
        return f"Exception: {str(e)}", 0.0

# Page config
st.set_page_config(page_title="Ollama Chat", layout="wide")
st.title("Chatlama - Chat with Ollama Models")

# Sidebar - Model selection
model_list = get_local_ollama_models()
if not model_list or "Error" in model_list[0] or "Exception" in model_list[0]:
    st.sidebar.error("🚫 No Ollama models installed or Ollama server is not running.")
    st.stop()
else:
    model = st.sidebar.selectbox("🔍 Choose Ollama Model", model_list)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "response_times" not in st.session_state:
    st.session_state.response_times = []

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0.0

if "model_used" not in st.session_state:
    st.session_state.model_used = ''

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.doc_texts = []
    st.session_state.embedder = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Chat Performance")

# File upload
uploaded_file = st.file_uploader("📄 Upload a Word Document (.docx)", type=["docx"])

doc_content = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    pages = load_docx_documents(tmp_path)
    split_docs = split_documents(pages)
    index, doc_texts, embedder = create_vector_store(split_docs)

    st.session_state.faiss_index = index
    st.session_state.doc_texts = doc_texts
    st.session_state.embedder = embedder

    st.success("✅ Document uploaded, processed, and indexed with FAISS!")

# Chat input
st.subheader("💬 Chat with the Model")
user_input = st.text_input("Enter your question:")

col1, col2 = st.columns([3, 1])

# Handle user input and response
with col1:
    if st.button("Send"):
        if user_input.strip() != "":
            if st.session_state.faiss_index and st.session_state.embedder:
                context = retrieve_context(user_input, st.session_state.embedder, st.session_state.faiss_index, st.session_state.doc_texts)
                context_str = "\n".join(context)
                prompt = f"Context:\n{context_str}\n\nQuestion: {user_input}"
            else:
                prompt = user_input

            with st.spinner(f"Getting response from model...{model}"):
                response, response_time = chat_with_model(model, prompt)

            st.session_state.chat_history.append((user_input, response, model))
            st.session_state.response_times.append(response_time)
            st.session_state.last_response_time = response_time
            st.session_state.model_used = model
        else:
            st.warning("❗ Please enter a question.")

# Display chat history
for question, answer, model_used in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {question}")
    st.markdown(f"**{model_used}:** {answer}")
    st.markdown("---")

# Metrics in sidebar
st.sidebar.markdown(f"Model: **{st.session_state.model_used}**")
st.sidebar.metric("Response Time (s)", value=str(st.session_state.last_response_time))
total_time = sum(st.session_state.response_times)
st.sidebar.metric("Total Time (s)", value=str(round(total_time, 2)))

# Get system details
cpu_info = platform.processor()
system_info = platform.system() + " " + platform.release()
memory = psutil.virtual_memory()
total_memory_gb = round(memory.total / (1024**3), 2)
available_memory_gb = round(memory.available / (1024**3), 2)
cpu_usage_percent = psutil.cpu_percent(interval=1)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**OS:** {system_info}")
st.sidebar.markdown(f"**CPU:** {cpu_info}")
st.sidebar.markdown(f"**CPU Usage:** {cpu_usage_percent}%")
st.sidebar.markdown(f"**Memory:** {available_memory_gb} GB / {total_memory_gb} GB")
