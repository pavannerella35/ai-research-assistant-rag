# AI Research Assistant — Multi-Agent Retrieval Q&A

A multi-agent question-answering assistant over any text knowledge base: a **Router agent** classifies the question, a **Retriever agent** pulls the most relevant passages via vector search, and a **Synthesizer agent** composes a grounded, cited answer. Same agent-orchestration and retrieval pattern used in production multi-agent GenAI systems (CrewAI/AutoGen-style role division), applied here to a general-purpose document set instead of any single employer's data.

## What it does

- Ingests any folder of `.txt` documents (sample: a small internal product-docs knowledge base), chunks them, and builds a TF-IDF vector index.
- Runs a 3-agent pipeline:
  1. **RouterAgent** — classifies the incoming question as `factual`, `comparison`, or `howto` to decide retrieval depth.
  2. **RetrieverAgent** — retrieves top-k relevant chunks by cosine similarity.
  3. **SynthesizerAgent** — composes an answer strictly from retrieved context and reports which sources it used.
- Falls back to extractive synthesis when no LLM API key is present; swaps in GPT-4 via LangChain when `OPENAI_API_KEY` is set (see `src/agents.py`).
- CLI and FastAPI interfaces.

## Why this project

Demonstrates agent-orchestration design (role separation, routing, retrieval-augmented synthesis) and vector-search fundamentals — the same building blocks behind production multi-agent GenAI systems, applied to a self-contained, original dataset.

## Tech stack

Python, scikit-learn (TF-IDF retrieval), a lightweight custom agent-orchestration layer (swappable for CrewAI/AutoGen/LangGraph), FastAPI.

## Project structure

```
project-1-ai-research-assistant/
├── data/sample_knowledge_base/   # Sample generic product-docs corpus (txt)
├── src/
│   ├── ingest.py                  # Chunking + TF-IDF index build
│   ├── agents.py                  # Router / Retriever / Synthesizer agents
│   ├── orchestrator.py            # Coordinates the 3-agent pipeline
│   └── app.py                     # FastAPI service
├── tests/test_pipeline.py
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python src/ingest.py
python src/orchestrator.py "How do I reset my device to factory settings?"
uvicorn src.app:app --reload --app-dir .
```

## Testing

```bash
python tests/test_pipeline.py
```
