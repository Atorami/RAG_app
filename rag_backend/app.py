from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag_logic import rag_pipeline, collection, embedding_model, model, tokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
async def get_rag_answer(request: QueryRequest):
    answer, docs, metas = rag_pipeline(request.query, collection, embedding_model, model, tokenizer)
    return {
        "answer": answer,
        "sources": metas
    }

