import os
import time
import tempfile
import streamlit as st
import numpy as np
import json
import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import fitz  
import io
import re
import hashlib


OLLAMA_BASE_URL = "https://ollama.thecit.in"  #Ollama endpoint
DEFAULT_MODEL = "llama3.2"
MODELS = ["llama3.2", "llama3.3", "deepseek-r1:70b", "phi4"]
CHROMA_PERSIST_DIRECTORY = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# session state for storing conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


#extract text from a PDF WITHOUT using multiprocessing
def extract_text_from_pdf(pdf_path):
    start_time = time.time()
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    st.info(f"PDF has {total_pages} pages. Starting extraction...")
    progress_bar = st.progress(0)
    
    all_text = []
    
    for i, page in enumerate(doc):
        try:
            # First try to extract text directly
            text = page.get_text()
            
            # If the page has very little text, it might be an image that needs OCR
            if len(text.strip()) < 50:  # Arbitrary threshold
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            
            all_text.append(text)
            
            # Update progress
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)
            
        except Exception as e:
            st.warning(f"Error processing page {i+1}: {str(e)}")
            all_text.append(f"Error processing page {i+1}: {str(e)}")
    
    combined_text = "\n\n".join(all_text)
    
    end_time = time.time()
    st.success(f"Extraction completed in {end_time - start_time:.2f} seconds")
    
    return combined_text


#text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return text_splitter.split_text(text)


# Simple tokenization 
def simple_tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words


# Simple TF-IDF-like embedding function that doesn't require external libraries
def simple_embedding(text, vocab_size=1000):
    # Get tokens
    tokens = simple_tokenize(text)
    
    # MD5 hash to create a deterministic but well-distributed embedding
    token_hashes = []
    for token in tokens:
        hash_obj = hashlib.md5(token.encode())
        hash_bytes = hash_obj.digest()
        for b in hash_bytes:
            token_hashes.append(b % 100)
    
    if len(token_hashes) > 100:
        token_hashes = token_hashes[:100]
    else:
        token_hashes.extend([0] * (100 - len(token_hashes)))
    
    return token_hashes


# to get context based on keyword search
def get_context(query, chunks, k=5):
    if not chunks:
        return "Error: No document has been loaded yet."
    
    try:
        # Simple keyword matching - count how many query words appear in each chunk
        query_tokens = set(simple_tokenize(query))
        
        # Score each chunk
        chunk_scores = []
        for chunk in chunks:
            chunk_tokens = set(simple_tokenize(chunk))
            # Count matching tokens
            score = len(query_tokens.intersection(chunk_tokens))
            chunk_scores.append(score)
        
        # Get top k chunks by score
        if max(chunk_scores) == 0:  # No direct matches
            # Fall back to chunks containing the most words
            top_indices = sorted(range(len(chunks)), 
                                key=lambda i: len(simple_tokenize(chunks[i])), 
                                reverse=True)[:k]
        else:
            top_indices = sorted(range(len(chunk_scores)), 
                                key=lambda i: chunk_scores[i], 
                                reverse=True)[:k]
        
        top_chunks = [chunks[i] for i in top_indices]
        context = "\n\n".join(top_chunks)
        
        return context
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


# Function to generate a response from the LLM
def generate_response(query, context, model_name=DEFAULT_MODEL, simplify=False):
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=model_name)
    
    if simplify:
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful legal assistant that explains complex legal concepts in simple terms.
            
            Here is the context from a legal document:
            {context}
            
            User question: {query}
            
            First, provide a direct and accurate answer based on the context.
            Then, explain the answer in simple terms that a non-legal professional can understand.
            Use analogies where appropriate to make complex concepts easier to grasp.
            """
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful legal assistant.
            
            Here is the context from a legal document:
            {context}
            
            User question: {query}
            
            Provide a direct and accurate answer based on the context. 
            If the answer cannot be found in the context, say so clearly.
            Do not make up information.
            """
        )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": context, "query": query})
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Streamlit UI
st.title("Legal Document Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Select AI Model", MODELS, index=MODELS.index(DEFAULT_MODEL))
    simplify_option = st.checkbox("Simplify legal language", value=False)
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and (not st.session_state.document_loaded or 
                         uploaded_file.name != st.session_state.document_name):
        st.session_state.document_loaded = False
        
        with st.spinner("Processing document..."):
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name
            
            # Extract text
            try:
                extracted_text = extract_text_from_pdf(pdf_path)
                
                # Split into chunks
                chunks = split_text(extracted_text)
                st.info(f"Document split into {len(chunks)} chunks")
                
                # Store chunks directly in session state
                st.session_state.chunks = chunks
                st.session_state.document_loaded = True
                st.session_state.document_name = uploaded_file.name
                st.success(f"Document '{uploaded_file.name}' loaded successfully!")
                
                # Clean up
                os.unlink(pdf_path)
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    if st.session_state.document_loaded:
        st.success(f"Active document: {st.session_state.document_name}")
    
    st.header("About")
    st.write("This application helps you analyze legal documents using AI.")

# Main chat interface
if not st.session_state.document_loaded:
    st.info("Please upload a document to begin.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new question
    user_input = st.chat_input("Ask a question about your document")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get context from chunks
                context = get_context(user_input, st.session_state.chunks)
                
                # Generate response
                response = generate_response(
                    user_input, 
                    context, 
                    model_name=selected_model,
                    simplify=simplify_option
                )
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})