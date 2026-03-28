from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


SECTION_REGEX = re.compile(r"^(\d+(\.\d+){0,3})\s+(.+)$")


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    section_title: str
    page_number: int
    text: str


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace so chunking logic is deterministic."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(page: PageText) -> List[tuple[str, str, str]]:
    """
    Split one page into section-aware blocks.
    Returns tuples: (section_id, section_title, section_text).
    """
    lines = [line.strip() for line in page.text.split("\n")]
    sections: List[tuple[str, str, str]] = []

    current_section_id = f"page-{page.page_number}"
    current_section_title = "Unlabeled Section"
    buffer: List[str] = []

    for line in lines:
        if not line:
            continue
        match = SECTION_REGEX.match(line)
        if match:
            # Flush content collected for the previous section before starting a new one.
            if buffer:
                sections.append(
                    (
                        current_section_id,
                        current_section_title,
                        normalize_whitespace("\n".join(buffer)),
                    )
                )
                buffer = []
            current_section_id = match.group(1)
            current_section_title = match.group(3).strip()
            continue
        buffer.append(line)

    if buffer:
        sections.append(
            (
                current_section_id,
                current_section_title,
                normalize_whitespace("\n".join(buffer)),
            )
        )

    return sections


def chunk_text(
    pages: List[PageText],
    chunk_size_chars: int = 900,
    overlap_chars: int = 180,
) -> List[Chunk]:
    """
    Create chunks that preserve section and page metadata.
    We slide by character window because PDF text line lengths vary heavily.
    """
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")

    chunks: List[Chunk] = []
    global_counter = 0

    for page in pages:
        page_sections = split_into_sections(page)
        for section_id, section_title, section_text in page_sections:
            if not section_text:
                continue

            start = 0
            while start < len(section_text):
                end = min(start + chunk_size_chars, len(section_text))
                chunk_body = section_text[start:end]
                chunk_id = f"chunk-{global_counter}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        section_id=section_id,
                        section_title=section_title,
                        page_number=page.page_number,
                        text=chunk_body,
                    )
                )
                global_counter += 1
                if end == len(section_text):
                    break
                start = end - overlap_chars

    return chunks
