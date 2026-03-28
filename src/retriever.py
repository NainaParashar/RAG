from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk


class ChunkRetriever:
    """
    TF-IDF retriever for chunk search.
    This keeps the solution local and lightweight, with no external model dependency.
    """

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        chunk_texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(chunk_texts)

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        """Return top chunks with score and metadata."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_idx:
            chunk = self.chunks[int(idx)]
            result = asdict(chunk)
            result["score"] = float(scores[int(idx)])
            results.append(result)

        return results
