# Agent-RAG (Retrieval-Augmented-Generation) API

Minimal FastAPI application that loads a web article, builds a Chroma vector store using Google embeddings, and answers user queries through a RAG agent.

## Requirements
- Python 3.14+
- `requirements.txt` dependencies installed
- A valid `GOOGLE_API_KEY` for Google Generative AI services

## Setup
1. Create a Python virtual environment:
```bash
python -m venv .venv
```
2. Activate the environment:
- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file in the project root with:
```env
LANGSMITH_TRACING ="true"
LANGSMITH_API_KEY ="<YOUR-API-KEY>"
GOOGLE_API_KEY="<YOUR-API-KEY>"
GOOGLE_CHAT_MODEL="gemini-3.1-flash-lite-preview"
GOOGLE_EMBEDDING_MODEL="models/gemini-embedding-001"
```

## Run the API
```bash
uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

## Endpoints
- `GET /health` — health check
- `GET /data` — fetch raw GitHub page content
- `POST /RagQuery` — submit JSON payload:
```json
{ "user_query": "Your question here" }
```

`curl-request`

```cuRL
curl -X POST "http://localhost:8000/RagQuery" -H "Content-Type: application/json" -d "{\"user_query\": \"What is the standard method for Task Decomposition and common extensions?\"}"
```

## Notes
- The application scrapes and indexes content from `https://lilianweng.github.io/posts/2023-06-23-agent/` on startup.
- The vector store is persisted under `./chroma_langchain_docs_db`.

## Overview Video

<video src="media/Overview.mp4" width="100%" controls autoplay loop muted>
  Your browser does not support the video tag.
</video>

