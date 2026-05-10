"""
FastAPI service for the SHL Assessment Recommender Agent.
Endpoints: GET /health, POST /chat
Loads catalog and FAISS index in a background thread so the port opens immediately.
"""
import asyncio
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retrieval import initialize_retriever


# --- Global readiness flag ---
_ready = False


def _background_init():
    """Run heavy initialization in a background thread."""
    global _ready
    print("[Startup] Background: Loading catalog and FAISS index...", flush=True)
    try:
        initialize_retriever()
        _ready = True
        print("[Startup] Background: Initialization complete. Ready to serve.", flush=True)
    except Exception as e:
        print(f"[Startup] Background: INIT FAILED: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background initialization, yield immediately so uvicorn binds the port."""
    thread = threading.Thread(target=_background_init, daemon=True)
    thread.start()
    print("[Startup] Server is starting, model loading in background...", flush=True)
    yield
    print("[Shutdown] Cleaning up...", flush=True)


# --- Pydantic models for strict schema validation ---

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: str


# --- FastAPI app ---

app = FastAPI(
    title="SHL Assessment Recommender Agent",
    description="A conversational AI agent that helps hiring managers find the right SHL assessments.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint. Responds immediately even during init."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for assessment recommendations.

    Accepts full conversation history and returns a response with
    optional recommendations.
    """
    # If the model isn't loaded yet, return a friendly message
    if not _ready:
        return ChatResponse(
            reply="I'm still warming up — please try again in a few seconds!",
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        # Lazy import to avoid circular issues during background init
        from agent import process_chat

        # Convert Pydantic models to dicts for the agent
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Process through the agent pipeline (with 25-second timeout)
        try:
            result = await asyncio.wait_for(
                process_chat(messages),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            # Return a graceful timeout response
            result = {
                "reply": "I'm taking a bit longer than expected. Could you rephrase your request or provide more specific details about the role?",
                "recommendations": [],
                "end_of_conversation": False,
            }

        # Build response with strict schema
        recommendations = [
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in result.get("recommendations", [])
        ]

        return ChatResponse(
            reply=result.get("reply", ""),
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )

    except Exception as e:
        print(f"[Error] Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Run directly with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
