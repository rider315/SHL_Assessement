"""
Vector store and retrieval system for SHL assessments.
Embeds assessment data using sentence-transformers and stores in FAISS.
Pre-loaded once at startup, never re-initialized per request.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Test type code to full name mapping
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Paths
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
VECTORS_DIR = os.path.join(os.path.dirname(__file__), "catalog_vectors")
INDEX_PATH = os.path.join(VECTORS_DIR, "faiss_index.bin")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class AssessmentRetriever:
    """Manages the FAISS vector store for SHL assessment retrieval."""

    def __init__(self):
        self.catalog: list[dict] = []
        self.index: faiss.IndexFlatIP | None = None
        self.model: SentenceTransformer | None = None
        self.url_set: set[str] = set()  # For URL validation

    def _build_embedding_text(self, item: dict) -> str:
        """Build a rich text representation for embedding an assessment."""
        parts = [item.get("name", "")]

        # Add full test type names
        test_types = item.get("test_type", [])
        if isinstance(test_types, list):
            type_names = [TEST_TYPE_MAP.get(t, t) for t in test_types]
            parts.append(" ".join(type_names))
        elif isinstance(test_types, str) and test_types:
            parts.append(TEST_TYPE_MAP.get(test_types, test_types))

        # Add description
        desc = item.get("description", "")
        if desc:
            parts.append(desc)

        # Add job levels for better matching
        job_levels = item.get("job_levels", "")
        if job_levels:
            parts.append(f"Job levels: {job_levels}")

        return " | ".join(filter(None, parts))

    def load(self):
        """Load catalog and build/load the FAISS index."""
        print("[Retrieval] Loading catalog...")
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        # Build URL validation set
        self.url_set = {item["url"] for item in self.catalog}
        print(f"[Retrieval] Loaded {len(self.catalog)} assessments")

        # Load embedding model
        print(f"[Retrieval] Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Try to load existing FAISS index
        if os.path.exists(INDEX_PATH):
            print("[Retrieval] Loading existing FAISS index...")
            self.index = faiss.read_index(INDEX_PATH)
            if self.index.ntotal == len(self.catalog):
                print(f"[Retrieval] FAISS index loaded ({self.index.ntotal} vectors)")
                return
            else:
                print("[Retrieval] Index size mismatch, rebuilding...")

        # Build new index
        self._build_index()

    def _build_index(self):
        """Build FAISS index from catalog embeddings."""
        print("[Retrieval] Building embeddings...")
        texts = [self._build_embedding_text(item) for item in self.catalog]
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Use Inner Product (cosine similarity since vectors are normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Save index
        os.makedirs(VECTORS_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        print(f"[Retrieval] FAISS index built and saved ({self.index.ntotal} vectors, dim={dim})")

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """Search for assessments matching the query."""
        if not self.model or not self.index:
            raise RuntimeError("Retriever not initialized. Call load() first.")

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        # Search FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.catalog)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.catalog):
                continue
            item = self.catalog[idx].copy()
            item["relevance_score"] = float(score)
            results.append(item)

        return results

    def validate_url(self, url: str) -> bool:
        """Check if a URL exists in the scraped catalog."""
        return url in self.url_set

    def get_assessment_by_name(self, name: str) -> dict | None:
        """Look up an assessment by exact or partial name match."""
        name_lower = name.lower()
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    def build_query_from_messages(self, messages: list[dict]) -> str:
        """Build a single search query from the full conversation history.
        Concatenates all user messages to capture the full context."""
        user_messages = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]
        return " ".join(user_messages)


# Module-level singleton
_retriever: AssessmentRetriever | None = None


def get_retriever() -> AssessmentRetriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = AssessmentRetriever()
        _retriever.load()
    return _retriever


def initialize_retriever():
    """Pre-initialize the retriever (call at app startup)."""
    get_retriever()
