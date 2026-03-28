from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rank_bm25 import BM25Okapi

from .smart_ingestion import SmartChunk


class HybridRetriever:
    """
    Phase 2 + 3 retrieval stack:
    - Dense retrieval (SentenceTransformer + Chroma)
    - Sparse retrieval (BM25)
    - Reciprocal Rank Fusion
    - Parent section expansion
    - Cross-reference expansion
    - Optional cross-encoder reranking
    """

    def __init__(
        self,
        chunks: list[SmartChunk],
        persist_dir: str = "data/chroma",
        light_mode: bool = False,
    ) -> None:
        self.chunks = chunks
        self.light_mode = light_mode
        self.chunk_by_id = {c.id: c for c in chunks}
        self.by_section = {}
        for c in chunks:
            self.by_section.setdefault(c.section_number, []).append(c)

        self.embedder = None
        self.reranker = None
        self.collection = None
        if not self.light_mode:
            # Lazy-import heavy libraries so light mode starts fast.
            import chromadb
            from sentence_transformers import CrossEncoder, SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.chroma = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma.get_or_create_collection(name="nasa_handbook")

            docs = [c.text for c in chunks]
            embeddings = self.embedder.encode(docs, convert_to_numpy=True).tolist()
            metadatas = []
            for c in chunks:
                m = {k: v for k, v in asdict(c).items() if k != "text" and v is not None}
                # ChromaDB rejects empty lists and complex arrays in metadata
                if "references" in m:
                    if not m["references"]:
                        del m["references"]
                    else:
                        m["references"] = ",".join(m["references"])
                metadatas.append(m)
            ids = [c.id for c in chunks]

            # Upsert ensures rebuilds remain idempotent.
            # Batch upsert to prevent chroma limit crash
            BATCH_SIZE = 5000
            for i in range(0, len(docs), BATCH_SIZE):
                self.collection.upsert(
                    documents=docs[i:i+BATCH_SIZE],
                    embeddings=embeddings[i:i+BATCH_SIZE],
                    metadatas=metadatas[i:i+BATCH_SIZE],
                    ids=ids[i:i+BATCH_SIZE],
                )

        self.bm25 = BM25Okapi([c.text.split() for c in chunks])

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _dense_retrieve(self, query: str, top_k: int) -> list[str]:
        if self.light_mode or self.embedder is None or self.collection is None:
            return []
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).tolist()
        res = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        return res["ids"][0] if res.get("ids") else []

    def _sparse_retrieve(self, query: str, top_k: int) -> list[str]:
        scored = self.bm25.get_scores(query.split())
        ranked = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        return [self.chunks[i].id for i in ranked]

    def _merge_with_rrf(self, dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for r, chunk_id in enumerate(dense_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + self._rrf(r)
        for r, chunk_id in enumerate(sparse_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + self._rrf(r)
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def _add_parent_sections(self, chunk_ids: list[str]) -> list[str]:
        expanded = list(chunk_ids)
        seen = set(expanded)
        queue = list(chunk_ids)
        
        while queue:
            cid = queue.pop(0)
            c = self.chunk_by_id[cid]
            if c.parent_section and c.parent_section in self.by_section:
                for parent_chunk in self.by_section[c.parent_section]:
                    if parent_chunk.id not in seen:
                        expanded.append(parent_chunk.id)
                        seen.add(parent_chunk.id)
                        queue.append(parent_chunk.id)  # recursively load parent's parents if needed
        return expanded

    def _resolve_cross_references(self, chunk_ids: list[str]) -> list[str]:
        expanded = list(chunk_ids)
        seen = set(expanded)
        for cid in chunk_ids:
            chunk = self.chunk_by_id[cid]
            for ref in chunk.references:
                for ref_chunk in self.by_section.get(ref, [])[:2]:
                    if ref_chunk.id not in seen:
                        expanded.append(ref_chunk.id)
                        seen.add(ref_chunk.id)
        return expanded

    def _rerank(self, query: str, chunk_ids: list[str], top_k: int) -> list[SmartChunk]:
        if self.light_mode or self.reranker is None:
            return [self.chunk_by_id[cid] for cid in chunk_ids[:top_k]]
        pairs = [[query, self.chunk_by_id[cid].text] for cid in chunk_ids]
        if not pairs:
            return []
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(chunk_ids, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [self.chunk_by_id[cid] for cid, _ in ranked[:top_k]]

    def retrieve(self, query: str, expanded_query: str, top_k: int = 8) -> list[SmartChunk]:
        sparse_ids = self._sparse_retrieve(expanded_query, top_k=top_k)
        if self.light_mode:
            merged = sparse_ids
        else:
            dense_ids = self._dense_retrieve(expanded_query, top_k=top_k)
            merged = self._merge_with_rrf(dense_ids, sparse_ids)
        with_parents = self._add_parent_sections(merged)
        with_refs = self._resolve_cross_references(with_parents)
        return self._rerank(query, with_refs, top_k=top_k)


def chunk_to_dict(chunk: SmartChunk) -> dict[str, Any]:
    return asdict(chunk)
