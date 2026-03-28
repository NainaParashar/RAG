# Problem 2 Solution: NASA Manual QA (5-Phase Architecture)

This implementation follows the interview plan with a structure-first ingestion and hybrid retrieval stack.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase Mapping

1. **Smart ingestion (`src/smart_ingestion.py`)**
   - PyMuPDF block parsing with font-size based heading detection
   - section hierarchy metadata (`section_number`, `parent_section`, page range)
   - table extraction via Camelot (graceful fallback if not available)
   - figure-awareness stub chunks
   - acronym map generation

2. **Hybrid vector store (`src/hybrid_retriever.py`)**
   - dense: SentenceTransformer embeddings + ChromaDB
   - sparse: BM25 index
   - merge: Reciprocal Rank Fusion (RRF)

3. **Multi-strategy retrieval (`src/hybrid_retriever.py`)**
   - acronym-expanded query
   - parent section expansion
   - cross-reference expansion from `see Section X.Y`
   - cross-encoder reranking

4. **Structured synthesis (`src/synthesis.py`)**
   - system prompt with strict grounded-citation instructions
   - GPT-4o synthesis when `OPENAI_API_KEY` is set
   - deterministic fallback summary for local/offline demo

5. **Demo UI (`streamlit_app.py`)**
   - PDF uploader + auto-build
   - answer panel
   - source panel with clickable page links
   - architecture diagram rendering when available

## CLI Usage

Build artifacts:
```bash
python app.py build --pdf "/absolute/path/to/nasa_systems_engineering_handbook_0.pdf"
```

Ask one question:
```bash
python app.py ask --question "How does risk management feed into technical reviews?"
```

Fast lightweight mode (BM25 only):
```bash
python app.py ask --question "How does risk management feed into technical reviews?" --light-mode
```

Interactive mode:
```bash
python app.py chat
```

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

In sidebar:
- upload PDF (or provide path)
- choose **Light mode (faster startup)** when you want quick demo performance
- optionally enable **Auto-build after upload**
- click **Build KB** or **Load Artifacts**

Then ask question in main panel.

## Free local LLM mode (no OpenAI billing)

If you have Ollama installed, the app can synthesize answers locally:

```bash
ollama pull llama3.2:3b
ollama serve
```

In another terminal:

```bash
cd /Users/nainaparashar/testing
source .venv/bin/activate
export OLLAMA_MODEL="llama3.2:3b"
streamlit run streamlit_app.py
```

Priority order used by app:
1. OpenAI (`OPENAI_API_KEY`)
2. Local Ollama (`OLLAMA_MODEL`, default `llama3.2:3b`)
3. Deterministic extractive fallback

## Key Artifacts

- `data/smart_chunks.json`
- `data/acronyms.json`
- `data/chroma/` (hybrid dense store)
