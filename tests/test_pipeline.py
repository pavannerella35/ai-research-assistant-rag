"""Basic smoke tests for the multi-agent research assistant pipeline."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ingest
from agents import RetrieverAgent, RouterAgent, SynthesizerAgent


def test_build_index_and_retrieve(tmp_path=None):
    ingest.build_index()
    assert os.path.exists(ingest.STORE_PATH)

    router = RouterAgent()
    retriever = RetrieverAgent()
    synthesizer = SynthesizerAgent()

    routed = router.route("How do I reset my device to factory settings?")
    assert routed["type"] == "howto"

    results = retriever.retrieve(routed)
    assert len(results) > 0
    assert any("device_setup" in r["source"] for r in results)

    answer = synthesizer.synthesize(routed, results)
    assert "reset" in answer.lower()
    print("test_build_index_and_retrieve passed")


def test_comparison_routing():
    router = RouterAgent()
    routed = router.route("What is the difference between Plus and Pro plans?")
    assert routed["type"] == "comparison"
    print("test_comparison_routing passed")


if __name__ == "__main__":
    test_build_index_and_retrieve()
    test_comparison_routing()
    print("All tests passed.")
