from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

from .smart_ingestion import SmartChunk


SYSTEM_PROMPT = """You are a NASA Systems Engineering expert assistant.
Answer questions using ONLY the provided context chunks.

If the provided context chunks DO NOT contain sufficient information to answer the user's question, you MUST immediately reply with the following template:
"Based on the NASA Systems Engineering Handbook, I could not find a definitive answer to your question. The most relevant information I found pertains to [briefly describe related topics found in context], but it does not directly address your specific query."

If you CAN answer the question, adhere strictly to these rules:
Every factual claim MUST be followed by [Section X.Y, Page N].
If you cannot find a specific section number in the context, say 
"I could not find a precise source for this" — do not invent one.
If the answer spans multiple sections, show how they connect.
If a diagram is relevant, mention it by figure number.
End with a confidence score (High/Medium/Low) based on how directly
the context addresses the question.

IMPORTANT STYLE RULES:
- Do NOT show your reasoning process.
- Do NOT use phrases like "let's think", "first", "next", "it is reasonable to infer".
- Keep answer to the point and concise.
- Output format must be:
  Answer: <2-4 short lines max>
  Confidence: <High/Medium/Low>"""


def format_chunks(chunks: list[SmartChunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[Section {c.section_number}, Page {c.page_start}, Type {c.chunk_type}] {c.text}"
        )
    return "\n\n".join(lines)


def synthesize_answer(query: str, chunks: list[SmartChunk]) -> str:
    """
    Phase 4 synthesis:
    - Uses GPT-4o if OPENAI_API_KEY is present.
    - Falls back to deterministic chunk summary for local/offline demos.
    """
    context = format_chunks(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Return concise final answer only (no reasoning), with in-line citations."
    )
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or "No answer generated."

    # Free local path via Ollama if available on localhost.
    # You can configure model with OLLAMA_MODEL; default is lightweight.
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{user_prompt}"
    )
    payload = {
        "model": ollama_model,
        "prompt": ollama_prompt,
        "stream": False,
    }

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = (body.get("response") or "").strip()
            if text:
                return text
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass

    # Safe final fallback when neither OpenAI nor Ollama is available.
    snippet = "\n".join(
        [
            f"- [{c.section_number}, p.{c.page_start}] {c.text[:180]}..."
            for c in chunks[:4]
        ]
    )
    return (
        "Neither OPENAI_API_KEY nor local Ollama response available, "
        "so showing grounded extractive summary:\n"
        f"{snippet}\n\nConfidence: Medium"
    )

def validate_citations(answer: str, retrieved_chunks: list[SmartChunk]) -> tuple[str, dict]:
    import re
    cited = re.findall(r'Section ([\d.]+)', answer)
    valid_sections = {c.section_number for c in retrieved_chunks}
    hallucinated = [s for s in cited if s not in valid_sections]
    if hallucinated:
        return answer, {"warning": f"Unverified citations: {hallucinated}"}
    return answer, {"status": "all citations verified"}


def answer_payload(query: str, chunks: list[SmartChunk]) -> dict[str, Any]:
    text = synthesize_answer(query, chunks)
    # Remove stray control characters commonly produced by PDF extraction.
    cleaned_text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    
    validated_text, verification = validate_citations(cleaned_text, chunks)

    return {
        "query": query,
        "answer": validated_text,
        "verification": verification,
        "sources": [
            {
                "id": c.id,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_type": c.chunk_type,
                "preview": "".join(ch for ch in c.text[:220] if ch.isprintable() or ch in "\n\t"),
            }
            for c in chunks
        ],
    }
