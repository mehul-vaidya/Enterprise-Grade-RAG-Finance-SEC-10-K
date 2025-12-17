# Step6_Backend/app.py
# Backend ONLY – uses Step 3 and Step 4 exactly as-is

#to run this use below file
#python -m uvicorn Step5_Backend.app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path

from Step3_Faiss_search.faiss_search import search
from Step4_LLM_Search.llm_generation import generate_answer

app = FastAPI(title="Financial RAG API")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: ChatRequest):
    # Step 3: FAISS retrieval + reranking
    # print("query " + req.query)
    reranked_chunks = search(req.query)
    # print("reranked_chunks " + reranked_chunks)
    # Step 4: LLM generation (local Ollama)
    result = generate_answer(req.query, reranked_chunks)
    # print("result " + result)
    return result


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return Path("Step5_Backend/ui.html").read_text(encoding="utf-8")
