
# Enterprise-Grade RAG Platform for Financial Data (SEC 10-K)

## Purpose
This project builds an end-to-end Retrieval-Augmented Generation (RAG) system for querying SEC 10-K filings.
It uses FAISS for vector search, Cross-Encoder reranking, and a local LLM (Ollama) for answer generation with citations.

## Architecture
SEC 10-K HTML → Ingestion → FAISS → Reranker → LLM → FastAPI → UI

## Requirements
- Python 3.10+
- Ollama (llama3 or mistral)
- FAISS
- sentence-transformers
- fastapi
- uvicorn
- beautifulsoup4
- tiktoken

## Setup

Create venv:
python -m venv .venv
.venv\Scripts\activate

Install deps:
pip install -r requirements.txt

Install Ollama:
https://ollama.com
ollama pull llama3
ollama serve

## Run

Ingestion (once):
python Step2_Ingestion/ingest.py

Backend:
python -m uvicorn Step5_Backend.app:app --reload

Open:
http://127.0.0.1:8000/docs

UI:
Open ui.html in browser

## Example Query
What are the main risk factors mentioned by Apple?
