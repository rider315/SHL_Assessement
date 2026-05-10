# SHL Assessment Recommender Agent

A production-grade conversational AI agent that helps hiring managers and recruiters find the right SHL assessments through natural dialogue.

## Architecture

```
User Request → FastAPI → Pre-LLM Guardrails → FAISS Retrieval → Gemini Flash LLM → Post-LLM Validation → Response
```

### Components
- **Scraper** (`scraper.py`): Scrapes all 32 pages of SHL Individual Test Solutions catalog
- **Retrieval** (`retrieval.py`): sentence-transformers/all-MiniLM-L6-v2 embeddings + FAISS vector store
- **Agent** (`agent.py`): Intent detection, Gemini 1.5 Flash LLM integration, response validation
- **API** (`main.py`): FastAPI with `/health` and `/chat` endpoints

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Scrape the catalog (one-time)
```bash
python scraper.py
```

### 4. Run the server
```bash
python main.py
# Or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need assessments for a senior Java developer"}]}'
```

## API Endpoints

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on Render
3. Set environment variable: `GEMINI_API_KEY`
4. Deploy with Docker

## Tech Stack
- **LLM**: Google Gemini 2.0 Flash (free tier)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Framework**: FastAPI + uvicorn
- **Scraping**: requests + BeautifulSoup4
