"""
Lightweight retrieval module using TF-IDF + cosine similarity.
Uses scikit-learn instead of PyTorch/sentence-transformers to stay under 512MB RAM.
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CATALOG_PATH = Path(__file__).parent / "catalog.json"

# Type code descriptions for enriching search text
TYPE_DESCRIPTIONS = {
    "A": "ability aptitude cognitive reasoning",
    "B": "biodata situational judgement",
    "C": "competencies behavioral",
    "D": "development 360 feedback",
    "E": "assessment exercises simulation",
    "K": "knowledge skills technical",
    "P": "personality behavior traits",
    "S": "simulations work sample",
}


class CatalogRetriever:
    """TF-IDF based retriever for the SHL assessment catalog."""

    def __init__(self):
        self.catalog = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._url_set = set()
        self._name_map = {}

    def initialize(self):
        """Load catalog and build TF-IDF index."""
        print("[Retrieval] Loading catalog...", flush=True)

        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        print(f"[Retrieval] Loaded {len(self.catalog)} assessments", flush=True)

        # Build lookup structures
        for item in self.catalog:
            self._url_set.add(item["url"])
            self._name_map[item["name"].lower()] = item

        # Build TF-IDF index
        print("[Retrieval] Building TF-IDF index...", flush=True)
        documents = []
        for item in self.catalog:
            # Combine all text fields into a single searchable document
            parts = [
                item.get("name", ""),
                item.get("name", ""),  # Boost name by repeating
                item.get("description", ""),
                item.get("job_levels", ""),
            ]

            # Add type descriptions
            test_types = item.get("test_type", [])
            if isinstance(test_types, list):
                for t in test_types:
                    parts.append(TYPE_DESCRIPTIONS.get(t, ""))
            elif isinstance(test_types, str):
                parts.append(TYPE_DESCRIPTIONS.get(test_types, ""))

            documents.append(" ".join(parts).lower())

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        print(f"[Retrieval] TF-IDF index built ({len(self.catalog)} docs, "
              f"{self.tfidf_matrix.shape[1]} features)", flush=True)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search catalog using TF-IDF cosine similarity."""
        if self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query.lower()])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:  # Minimum relevance threshold
                item = self.catalog[idx].copy()
                item["relevance_score"] = float(scores[idx])
                results.append(item)

        return results

    def validate_url(self, url: str) -> bool:
        """Check if a URL exists in our catalog."""
        return url in self._url_set

    def get_assessment_by_name(self, name: str) -> dict | None:
        """Find an assessment by name (case-insensitive)."""
        return self._name_map.get(name.lower())

    def build_query_from_messages(self, messages: list[dict]) -> str:
        """Extract a search query from conversation history."""
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        return " ".join(user_messages[-3:])  # Use last 3 user messages


# --- Singleton pattern ---
_retriever: CatalogRetriever | None = None


def initialize_retriever():
    """Initialize the global retriever singleton."""
    global _retriever
    _retriever = CatalogRetriever()
    _retriever.initialize()


def get_retriever() -> CatalogRetriever:
    """Get the global retriever instance."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")
    return _retriever
