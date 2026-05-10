"""
FastAPI service for the SHL Assessment Recommender Agent.
Endpoints: GET /health, POST /chat
Pre-loads catalog and FAISS index at startup via lifespan.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retrieval import initialize_retriever
from agent import process_chat


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


# --- FastAPI app with lifespan for startup initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector store and catalog at startup."""
    print("[Startup] Pre-loading catalog and FAISS index...")
    # Run the heavy initialization in a thread to not block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_retriever)
    print("[Startup] Initialization complete. Ready to serve requests.")
    yield
    print("[Shutdown] Cleaning up...")


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
    """Health check endpoint. Must respond within 2 minutes on cold start."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for assessment recommendations.

    Accepts full conversation history and returns a response with
    optional recommendations.
    """
    try:
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
