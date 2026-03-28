from __future__ import annotations

import argparse
import json

from src.pipeline import QAPipeline


def cmd_build(pdf_path: str, light_mode: bool) -> None:
    """Build artifacts using smart ingestion + hybrid retrieval pipeline."""
    pipe = QAPipeline(light_mode=light_mode)
    stats = pipe.build(pdf_path)
    print(
        f"Built pipeline artifacts: {stats['chunks']} chunks, "
        f"{stats['acronyms']} acronyms. "
        f"Mode: {'light' if stats['light_mode'] else 'full'}."
    )


def cmd_ask(question: str, top_k: int, light_mode: bool) -> None:
    """Ask a question against built artifacts."""
    pipe = QAPipeline(light_mode=light_mode)
    result = pipe.ask(question, top_k=top_k)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for c in result["sources"][:5]:
        print(
            f"- Section {c['section_number']} ({c['section_title']}) | "
            f"Page {c['page_start']} | Type {c['chunk_type']}"
        )
    print("\nRaw JSON:")
    print(json.dumps(result, indent=2))


def cmd_chat(top_k: int, light_mode: bool) -> None:
    """Interactive mode for live demo."""
    pipe = QAPipeline(light_mode=light_mode)
    pipe.load()
    print("Chat mode ready. Type 'exit' to quit.\n")
    while True:
        q = input("Question> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        result = pipe.ask(q, top_k=top_k)
        print("\nAnswer:")
        print(result["answer"])
        print("\nTop sources:")
        for c in result["sources"][:4]:
            print(f"- Page {c['page_start']} | Section {c['section_number']}")
        print("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Technical Manual QA System (smart hybrid pipeline)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build smart index from PDF")
    build.add_argument("--pdf", required=True, help="Path to source PDF")
    build.add_argument("--light-mode", action="store_true", help="Use BM25-only fast mode")

    ask = sub.add_parser("ask", help="Ask one question")
    ask.add_argument("--question", required=True, help="Natural language question")
    ask.add_argument("--top-k", type=int, default=8, help="Top chunks to retrieve")
    ask.add_argument("--light-mode", action="store_true", help="Use BM25-only fast mode")

    chat = sub.add_parser("chat", help="Start interactive QA loop")
    chat.add_argument("--top-k", type=int, default=8, help="Top chunks to retrieve")
    chat.add_argument("--light-mode", action="store_true", help="Use BM25-only fast mode")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args.pdf, args.light_mode)
    elif args.command == "ask":
        cmd_ask(args.question, args.top_k, args.light_mode)
    elif args.command == "chat":
        cmd_chat(args.top_k, args.light_mode)


if __name__ == "__main__":
    main()
