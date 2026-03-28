from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .chunker import Chunk


def save_chunks(chunks: List[Chunk], output_path: str) -> None:
    """Persist chunk metadata/text so ask mode is fast after first build."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_chunks(input_path: str) -> List[Chunk]:
    """Load serialized chunks from disk."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {input_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**row) for row in data]
