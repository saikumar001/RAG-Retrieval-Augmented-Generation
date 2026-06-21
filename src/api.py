import os
from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from pydantic import BaseModel, ConfigDict

# Import the refactored RAG pipeline layout logic
import rag

class RagQuery(BaseModel):
    model_config= ConfigDict(extra="forbid")
    user_query: str


# ============================================================
# FASTAPI SERVER LIFESPAN (Runs once during web context boot up)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes vector database contexts cleanly before processing HTTP endpoint queries."""
    print("[*] Initializing RAG background pipeline dependencies...")
    rag.load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("[-] Critical Configuration Error: GOOGLE_API_KEY configuration variable is missing.")
        raise RuntimeError("Missing GOOGLE_API_KEY context boundary conditions.")
        
    # Phase 1 setup runs exactly once when the API app starts
    raw_docs = rag.load_and_clean_web_content(rag.BLOG_URL)
    split_chunks = rag.chunk_documents(raw_docs)
    
    # Store instance directly inside global namespace pointer of your import module
    rag.vector_store = rag.initialize_vector_store(split_chunks)
    print("[✓] RAG Service Component initialized. Ready to receive POST requests.")
    yield
    print("[-] Terminating web service application instances.")


# Instantiating server context coupled with the lifespan hook
app = FastAPI(lifespan=lifespan)
router = APIRouter()


# ============================================================
# ENDPOINT LAYOUT ROUTING
# ============================================================

@router.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})


@router.get("/data")
async def get_data():
    try:
        data = await fetch_data("https://github.com/saikumar001/RAG-Retrieval-Augmented-Generation")
        return JSONResponse(content={"raw_content": data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_data(url: str):
    async with AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


@router.post("/RagQuery")
def user_query_endpoint(query: RagQuery):
    """Executes multi-step agent graphs leveraging cached vector database contexts directly."""
    if not query.user_query.strip():
        raise HTTPException(status_code=400, detail="The input 'user_query' parameter cannot be blank.")
        
    try:
        # Call the workflow execution loop directly (skipping the heavy ingestion stage)
        result = rag.execute_agentic_workflow(query.user_query)
        return JSONResponse(content={
            "query": query.user_query,
            "response": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")


app.include_router(router)