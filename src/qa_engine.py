from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from .retriever import ChunkRetriever


ACRONYM_MAP = {
    "trl": "technology readiness level",
    "kdp": "key decision point",
    "srr": "system requirements review",
    "pdr": "preliminary design review",
    "cdr": "critical design review",
}


def expand_acronyms(query: str) -> str:
    """
    Expand common NASA handbook acronyms into full forms
    to improve retrieval recall.
    """
    words = query.split()
    expanded: List[str] = []
    for w in words:
        key = re.sub(r"[^a-zA-Z]", "", w).lower()
        expanded.append(w)
        if key in ACRONYM_MAP:
            expanded.append(f"({ACRONYM_MAP[key]})")
    return " ".join(expanded)


def summarize_from_chunks(query: str, results: List[Dict]) -> str:
    """
    Small extractive summarizer:
    - picks relevant sentences from top chunks
    - combines them into a concise answer
    This is transparent and easy to explain in interviews.
    """
    query_terms = {
        t.lower()
        for t in re.findall(r"[A-Za-z]{3,}", query)
        if t.lower() not in {"what", "which", "when", "where", "how", "does"}
    }

    candidate_sentences: List[tuple[float, str]] = []
    for row in results[:4]:
        sentences = re.split(r"(?<=[.!?])\s+", row["text"])
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 40:
                continue
            sent_terms = set(re.findall(r"[A-Za-z]{3,}", s.lower()))
            overlap = len(query_terms.intersection(sent_terms))
            score = overlap + (row["score"] * 2.0)
            candidate_sentences.append((score, s))

    if not candidate_sentences:
        return "I could not find a reliable answer in the indexed content."

    candidate_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence in candidate_sentences:
        # Basic deduplication: avoid repeating near-identical lines.
        key = sentence.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) == 3:
            break

    return " ".join(selected)


def _extract_topic_keywords(query: str) -> List[str]:
    """Extract meaningful topic words from query for precision filtering."""
    stop = {
        "what",
        "which",
        "when",
        "where",
        "how",
        "does",
        "about",
        "there",
        "tell",
        "me",
        "is",
        "are",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "to",
    }
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", query)]
    return [t for t in tokens if t not in stop]


def _topic_focused_answer(query: str, results: List[Dict]) -> str | None:
    """
    For 'about X' style questions, return short bullet points that directly
    mention topic keywords. If no mention is found, return a clear fallback.
    """
    keywords = _extract_topic_keywords(query)
    if not keywords:
        return None

    matched_sentences: List[str] = []
    for row in results:
        for sentence in re.split(r"(?<=[.!?])\s+", row["text"]):
            s = sentence.strip()
            if len(s) < 20:
                continue
            s_lower = s.lower()
            if any(k in s_lower for k in keywords):
                matched_sentences.append(s)

    if not matched_sentences:
        key_text = ", ".join(keywords[:3])
        return f"No explicit mention found for: {key_text}."

    unique = []
    seen = set()
    for s in matched_sentences:
        k = s.lower()[:140]
        if k in seen:
            continue
        seen.add(k)
        unique.append(s)
        if len(unique) == 3:
            break

    return "\n".join([f"- {line}" for line in unique])


def _count_problem_statements_if_applicable(query: str, retriever: ChunkRetriever) -> Dict | None:
    """
    Handle count-style questions deterministically by scanning all indexed chunks.
    This avoids retrieval misses for simple structural questions.
    """
    q = query.lower()
    if "how many" not in q or "problem statement" not in q:
        return None

    pattern = re.compile(r"problem statement\s+(\d+)", re.IGNORECASE)
    found_numbers = set()
    pages_by_number: dict[str, set[int]] = defaultdict(set)

    for chunk in retriever.chunks:
        for match in pattern.findall(chunk.text):
            found_numbers.add(match)
            pages_by_number[match].add(chunk.page_number)

    if not found_numbers:
        return {
            "answer": "I could not find any explicit 'Problem Statement X' entries in the indexed document.",
            "citations": [],
        }

    sorted_numbers = sorted(found_numbers, key=lambda x: int(x))
    answer = (
        f"There are {len(sorted_numbers)} problem statements "
        f"(Problem Statement {' ,'.join(sorted_numbers)})."
    ).replace(" ,", ", ")

    citations = []
    for number in sorted_numbers[:4]:
        first_page = min(pages_by_number[number])
        citations.append(
            {
                "page": first_page,
                "section": f"problem-statement-{number}",
                "title": f"Problem Statement {number}",
                "score": 1.0,
                "preview": f"Problem Statement {number}",
            }
        )

    return {"answer": answer, "citations": citations}


def answer_question(query: str, retriever: ChunkRetriever, top_k: int = 6) -> Dict:
    """
    Generate final answer with citations from retrieved chunks.
    """
    deterministic = _count_problem_statements_if_applicable(query, retriever)
    if deterministic is not None:
        return {
            "query": query,
            "expanded_query": query,
            "answer": deterministic["answer"],
            "citations": deterministic["citations"],
        }

    expanded_query = expand_acronyms(query)
    results = retriever.retrieve(expanded_query, top_k=top_k)

    # If retrieval confidence is near-zero, avoid hallucinated long outputs.
    max_score = max((row["score"] for row in results), default=0.0)
    topic_answer = _topic_focused_answer(query, results)
    if max_score < 0.01 and topic_answer and topic_answer.startswith("No explicit mention found"):
        answer = topic_answer
    elif topic_answer and ("about " in query.lower() or "tell me" in query.lower()):
        answer = topic_answer
    else:
        answer = summarize_from_chunks(expanded_query, results)

    citations = []
    for row in results[:4]:
        citations.append(
            {
                "page": row["page_number"],
                "section": row["section_id"],
                "title": row["section_title"],
                "score": round(row["score"], 4),
                "preview": row["text"][:200].replace("\n", " "),
            }
        )

    return {
        "query": query,
        "expanded_query": expanded_query,
        "answer": answer,
        "citations": citations,
    }
