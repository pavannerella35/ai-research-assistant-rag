"""
app.py
FastAPI service exposing the multi-agent research assistant.
Run: uvicorn src.app:app --reload --app-dir .
Then: POST /ask {"question": "..."}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator import ResearchAssistant

app = FastAPI(title="AI Research Assistant", version="1.0.0")
assistant = ResearchAssistant()


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    return assistant.ask(payload.question)
