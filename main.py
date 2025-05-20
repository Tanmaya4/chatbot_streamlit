"""
RAG Chatbot with FAISS Retrieval and OpenRouter API-based LLM.

Dependencies:
- sentence-transformers
- faiss-cpu
- pandas
- requests
- streamlit (optional)

Author: Tanmaya
"""

import os
import faiss
import requests
import pandas as pd
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENROUTER_API_KEY = "APIKEY"
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"
DATA_FILE = "your_dataset.csv"

# Load and split documents
def load_and_split_documents(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    loader = DataFrameLoader(df, page_content_column="content")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    return [doc.page_content for doc in split_docs]

# Setup FAISS index
def setup_faiss(texts: List[str]):
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    return faiss_index, embedder, texts

# Semantic Retrieval using FAISS
def semantic_retrieve(query: str, faiss_index, embedder, texts, k: int = 5) -> List[str]:
    query_embedding = embedder.encode([query])
    faiss.normalize_L2(query_embedding)
    _, indices = faiss_index.search(np.array(query_embedding), k)
    return [texts[i] for i in indices[0]]

# Call OpenRouter API
def call_openrouter(messages: List[dict], model: str = OPENROUTER_MODEL) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    if not response.ok:
        print("OpenRouter API error:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# Chat function (RAG only, no memory)
def chat_with_rag(user_input: str, faiss_index, embedder, texts) -> str:
    context_docs = semantic_retrieve(user_input, faiss_index, embedder, texts)
    context_text = "\n".join(context_docs)
    system_prompt = (
        f"You are a helpful assistant. Use the following knowledge base context to answer the question.\n\n"
        f"Context:\n{context_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    return call_openrouter(messages)

# CLI Chat Loop
def cli_chat_loop():
    print("💬 FAISS RAG Chatbot (type 'exit' to quit)\n")
    texts = load_and_split_documents(DATA_FILE)
    faiss_index, embedder, docs = setup_faiss(texts)
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = chat_with_rag(user_input, faiss_index, embedder, docs)
        print("AI:", response)

# Optional Streamlit App
def streamlit_app():
    import streamlit as st
    st.title("💬 FAISS RAG Chatbot (OpenRouter)")

    if "faiss_index" not in st.session_state:
        texts = load_and_split_documents(DATA_FILE)
        faiss_index, embedder, docs = setup_faiss(texts)
        st.session_state.faiss_index = faiss_index
        st.session_state.embedder = embedder
        st.session_state.docs = docs  

    user_input = st.text_input("Ask your question:")
    if user_input:
        response = chat_with_rag(
            user_input,
            st.session_state.faiss_index,
            st.session_state.embedder,
            st.session_state.docs  
        )
        st.markdown("**Response:**")
        st.write(response)

# Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FAISS RAG chatbot with OpenRouter.")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="Run mode: cli or web")
    args = parser.parse_args()
    if args.mode == "web":
        streamlit_app()
    else:
        cli_chat_loop()
