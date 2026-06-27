"""
orchestrator.py
Coordinates the Router -> Retriever -> Synthesizer agent pipeline.
Run directly for a CLI demo: `python src/orchestrator.py "<question>"`
"""
import sys

from agents import RetrieverAgent, RouterAgent, SynthesizerAgent


class ResearchAssistant:
    def __init__(self):
        self.router = RouterAgent()
        self.retriever = RetrieverAgent()
        self.synthesizer = SynthesizerAgent()

    def ask(self, question: str) -> dict:
        routed = self.router.route(question)
        results = self.retriever.retrieve(routed)
        answer = self.synthesizer.synthesize(routed, results)
        return {
            "question": question,
            "question_type": routed["type"],
            "answer": answer,
            "sources_consulted": sorted({r["source"] for r in results}),
        }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I reset my device to factory settings?"
    assistant = ResearchAssistant()
    result = assistant.ask(question)
    print(f"Q: {result['question']}")
    print(f"[routed as: {result['question_type']}]\n")
    print(result["answer"])
    print(f"\nSources consulted: {', '.join(result['sources_consulted'])}")
