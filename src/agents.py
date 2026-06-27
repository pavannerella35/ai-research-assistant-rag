"""
agents.py
Three cooperating agents, each with a single responsibility -- the same role
-separation pattern used by frameworks like CrewAI/AutoGen/LangGraph, written
here as plain classes so the pipeline has zero required external services.

  RouterAgent       -> classifies question type, decides how many chunks to pull
  RetrieverAgent     -> vector search over the TF-IDF index (see ingest.py)
  SynthesizerAgent   -> composes the final, cited answer from retrieved context
"""
import os
import pickle
import re

from sklearn.metrics.pairwise import cosine_similarity

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store.pkl")
GENERATE_WITH_LLM = bool(os.environ.get("OPENAI_API_KEY"))


class RouterAgent:
    """Classifies the question so downstream agents can adjust behavior."""

    COMPARISON_MARKERS = ("difference", "versus", " vs ", "compare", "better")
    HOWTO_MARKERS = ("how do i", "how to", "steps to", "reset", "set up", "setup")

    def route(self, question: str) -> dict:
        q = question.lower()
        if any(marker in q for marker in self.COMPARISON_MARKERS):
            qtype, top_k = "comparison", 4
        elif any(marker in q for marker in self.HOWTO_MARKERS):
            qtype, top_k = "howto", 2
        else:
            qtype, top_k = "factual", 3
        return {"question": question, "type": qtype, "top_k": top_k}


class RetrieverAgent:
    """Retrieves the most relevant chunks for a routed question."""

    def __init__(self, store_path: str = STORE_PATH):
        if not os.path.exists(store_path):
            raise FileNotFoundError(
                f"No vector store found at {store_path}. Run `python src/ingest.py` first."
            )
        with open(store_path, "rb") as f:
            self.index = pickle.load(f)

    def retrieve(self, routed: dict) -> list[dict]:
        query_vec = self.index["vectorizer"].transform([routed["question"]])
        scores = cosine_similarity(query_vec, self.index["vectors"]).flatten()
        ranked = scores.argsort()[::-1][: routed["top_k"]]
        return [
            {
                "chunk": self.index["chunks"][i],
                "source": self.index["metadata"][i]["source"],
                "score": float(scores[i]),
            }
            for i in ranked
            if scores[i] > 0
        ]


class SynthesizerAgent:
    """Composes the final answer strictly from retrieved context."""

    def synthesize(self, routed: dict, results: list[dict]) -> str:
        if not results:
            return "I couldn't find anything relevant to that question in the knowledge base."
        if GENERATE_WITH_LLM:
            return self._llm_synthesize(routed, results)
        return self._extractive_synthesize(routed, results)

    def _extractive_synthesize(self, routed: dict, results: list[dict]) -> str:
        query_terms = set(re.findall(r"\w+", routed["question"].lower()))
        best_sentence, best_overlap, best_source = "", -1, None
        for result in results:
            for sentence in re.split(r"(?<=[.!?])\s+", result["chunk"]):
                overlap = len(query_terms & set(re.findall(r"\w+", sentence.lower())))
                if overlap > best_overlap:
                    best_sentence, best_overlap, best_source = sentence.strip(), overlap, result["source"]
        sources = ", ".join(sorted({r["source"] for r in results}))
        return f"{best_sentence}\n\n(Primary source: {best_source}; consulted: {sources})"

    def _llm_synthesize(self, routed: dict, results: list[dict]) -> str:
        """Production path: swap in LangChain + GPT-4, same retrieved context."""
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        context = "\n\n".join(f"[{r['source']}]\n{r['chunk']}" for r in results)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer using ONLY the provided excerpts. Cite sources. "
                           "If the answer isn't in the excerpts, say so."),
                ("human", "Excerpts:\n{context}\n\nQuestion type: {qtype}\nQuestion: {question}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = prompt | llm
        return chain.invoke(
            {"context": context, "qtype": routed["type"], "question": routed["question"]}
        ).content
