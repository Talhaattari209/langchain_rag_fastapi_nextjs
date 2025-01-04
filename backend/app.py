from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rag_pipeline import initialize_rag, query_rag

app = FastAPI()

# Initialize the RAG pipeline
rag_pipeline = initialize_rag()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_rag_api(request: QueryRequest):
    try:
        response = query_rag(rag_pipeline, request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
