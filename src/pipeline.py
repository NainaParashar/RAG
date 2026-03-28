from __future__ import annotations

from .hybrid_retriever import HybridRetriever
from .smart_ingestion import (
    SmartChunk,
    ingest_pdf_smart,
    load_ingestion_artifacts,
    save_ingestion_artifacts,
)
from .synthesis import answer_payload
import json
import os
import urllib.request
import urllib.error

CHUNKS_PATH = "data/smart_chunks.json"
ACRONYM_PATH = "data/acronyms.json"


def expand_acronyms(query: str, acronym_map: dict[str, str]) -> str:
    words = query.split()
    expanded = []
    for w in words:
        clean = "".join(ch for ch in w if ch.isalnum()).upper()
        expanded.append(w)
        if clean in acronym_map:
            expanded.append(f"({acronym_map[clean]})")
    return " ".join(expanded)


def decompose_query(query: str) -> list[str]:
    prompt = f"""Break this question into 2-3 sub-questions for document retrieval:
    Question: {query}
    Return as JSON list."""
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            pass
            
    # Free local path via Ollama if available
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = (body.get("response") or "").strip()
            if text:
                return json.loads(text)
    except Exception:
        pass
    
    return [query]


class QAPipeline:
    def __init__(self, light_mode: bool = False) -> None:
        self.light_mode = light_mode
        self.chunks: list[SmartChunk] = []
        self.acronym_map: dict[str, str] = {}
        self.retriever: HybridRetriever | None = None

    def build(self, pdf_path: str) -> dict:
        chunks, acronym_map = ingest_pdf_smart(pdf_path)
        save_ingestion_artifacts(chunks, acronym_map, CHUNKS_PATH, ACRONYM_PATH)
        self.chunks = chunks
        self.acronym_map = acronym_map
        self.retriever = HybridRetriever(chunks, light_mode=self.light_mode)
        return {
            "chunks": len(chunks),
            "acronyms": len(acronym_map),
            "light_mode": self.light_mode,
        }

    def load(self) -> dict:
        chunks, acronym_map = load_ingestion_artifacts(CHUNKS_PATH, ACRONYM_PATH)
        self.chunks = chunks
        self.acronym_map = acronym_map
        self.retriever = HybridRetriever(chunks, light_mode=self.light_mode)
        return {
            "chunks": len(chunks),
            "acronyms": len(acronym_map),
            "light_mode": self.light_mode,
        }

    def ask(self, query: str, top_k: int = 8) -> dict:
        if self.retriever is None:
            self.load()
        assert self.retriever is not None
        sub_queries = decompose_query(query)
        # Ensure the original query is always considered
        if query not in sub_queries:
            sub_queries.insert(0, query)
            
        all_chunks = []
        seen = set()
        
        # Retrieve chunks for each generated subquery
        for sq in sub_queries:
            expanded_sq = expand_acronyms(sq, self.acronym_map)
            # Use smaller top_k for subqueries to avoid context bloating
            for chunk in self.retriever.retrieve(query=sq, expanded_query=expanded_sq, top_k=top_k):
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    all_chunks.append(chunk)

        expanded_query = expand_acronyms(query, self.acronym_map)
        payload = answer_payload(query, all_chunks)
        payload["expanded_query"] = expanded_query
        return payload

