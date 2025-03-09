import io
import os
import time
import tempfile
import streamlit as st
import requests
import json
import fitz
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Dynamically fetch the Ngrok URL
def get_ngrok_url():
    try:
        response = requests.get("http://127.0.0.1:4040/api/tunnels")
        response.raise_for_status()
        tunnels = response.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "https":
                return tunnel.get("public_url")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Ngrok URL: {e}")
    return None

OLLAMA_BASE_URL = get_ngrok_url()
if not OLLAMA_BASE_URL:
    st.error("Ngrok is not running. Please check your Ngrok process.")
    st.stop()

DEFAULT_MODEL = "llama3.2"
MODELS = ["llama3.2"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    total_pages = len(doc)
    progress_bar = st.progress(0)

    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < 50:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
        all_text.append(text)
        progress_bar.progress((i + 1) / total_pages)

    progress_bar.empty()
    return "\n\n".join(all_text)


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return text_splitter.split_text(text)


def simple_tokenize(text):
    return text.lower().split()


def get_context(query, chunks, k=5):
    query_tokens = set(simple_tokenize(query))
    chunk_scores = [len(query_tokens.intersection(set(simple_tokenize(chunk)))) for chunk in chunks]
    top_indices = sorted(range(len(chunk_scores)), key=lambda i: chunk_scores[i], reverse=True)[:k]
    return "\n\n".join(chunks[i] for i in top_indices)


def generate_response(query, context, model_name=DEFAULT_MODEL, simplify=False):
    headers = {"Content-Type": "application/json"}

    if simplify:
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful legal assistant.

            Here is the context from a legal document:
            {context}

            User question: {query}

            First, provide a direct and accurate answer based on the context.
            Then, explain the answer in simple terms that a non-legal professional can understand.
            Use analogies where appropriate."""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful legal assistant.

            Here is the context from a legal document:
            {context}

            User question: {query}"""
        )

    prompt_text = prompt.format(context=context, query=query)

    payload = {
        "model": model_name,
        "prompt": prompt_text
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, headers=headers, stream=True)
        response.raise_for_status()
        full_response = "".join(json.loads(line).get("response", "") for line in response.iter_lines() if line)
        return full_response or "⚠️ No response received from Ollama API."
    except requests.exceptions.RequestException as e:
        return f"❌ Error: Unable to connect to Ollama API: {e}"

# Streamlit UI
st.title("Legal Document Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    selected_model = "llama3.2"
    simplify_option = st.checkbox("Simplify legal language", value=False)
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and (not st.session_state.document_loaded or 
                         uploaded_file.name != st.session_state.document_name):
        st.session_state.document_loaded = False
        
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name
            
            extracted_text = extract_text_from_pdf(pdf_path)
            chunks = split_text(extracted_text)
            st.session_state.chunks = chunks
            st.session_state.document_loaded = True
            st.session_state.document_name = uploaded_file.name
            st.success(f"Document '{uploaded_file.name}' loaded successfully!")
            
            os.unlink(pdf_path)
    
if not st.session_state.document_loaded:
    st.info("Please upload a document to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("Ask a question about your document")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = get_context(user_input, st.session_state.chunks)
                response = generate_response(user_input, context, selected_model, simplify_option)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
