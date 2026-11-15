# Resume ↔ Job Description Matcher (RAG)

A beginner–intermediate project that teaches **RAG, embeddings, semantic similarity, and structured output** by matching a resume against one or more job descriptions.

## What it does

1. Upload / paste a resume (PDF, DOCX, or TXT)
2. Compare it with one or more job descriptions
3. Retrieve **matching**, **partial**, and **missing** skills with resume evidence
4. Suggest resume improvements and likely interview questions

## How the RAG pipeline works

```
Resume  → chunk by section → embed → ChromaDB collection
JD      → extract requirements (LLM or keyword fallback)
For each requirement:
    hybrid retrieval (vector similarity + keyword) over resume chunks
    → matched / partial / missing
LLM (optional) → grounded structured analysis (Pydantic JSON)
```

| Concept | Where it lives |
|---|---|
| Chunking | `rag/chunking.py` |
| Embeddings | `rag/embeddings.py` (ONNX MiniLM, no PyTorch) |
| Vector store | `rag/vector_store.py` (ChromaDB) |
| Hybrid retrieval + scoring | `rag/pipeline.py` |
| Structured output schemas | `rag/schemas.py` |
| LLM client | `rag/llm.py` |

## Setup

```powershell
cd C:\Users\HP\Documents\Rag_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional LLM (copy `.env.example` → `.env`):

- OpenAI, Groq, or a local Ollama server
- Without an LLM the app still runs using keywords + embeddings + templates

The first run downloads the ~80 MB `all-MiniLM-L6-v2` ONNX model into `%USERPROFILE%\.cache\chroma\`.

## Run

**Web UI**

```powershell
.\.venv\Scripts\streamlit run app.py
```

**CLI** (sample data included)

```powershell
.\.venv\Scripts\python cli.py data\sample_resume.txt data\jd_ml_engineer.txt data\jd_frontend_developer.txt --no-llm
```

## Project layout

```
Rag_project/
  app.py                 # Streamlit UI
  cli.py                 # Command-line runner
  requirements.txt
  .env.example
  data/                  # Sample resume + 2 job descriptions
  rag/
    config.py
    schemas.py           # Pydantic structured output
    document_loader.py
    chunking.py
    embeddings.py
    vector_store.py
    skills.py            # Taxonomy + keyword fallback
    llm.py
    pipeline.py          # End-to-end RAG
```
