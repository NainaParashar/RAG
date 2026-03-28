from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from .chunker import PageText


def load_pdf_pages(pdf_path: str) -> List[PageText]:
    """
    Extract raw text page-by-page from a PDF.
    Keeping page boundaries is important so we can cite exact source pages later.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    pages: List[PageText] = []

    for idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append(PageText(page_number=idx, text=extracted))

    return pages
