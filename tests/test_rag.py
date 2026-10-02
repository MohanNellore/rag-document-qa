import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import RAGPipeline  # noqa: E402
from rag.chunker import chunk_text  # noqa: E402

DOCS = str(Path(__file__).resolve().parents[1] / "data" / "sample_docs")


class ChunkerTests(unittest.TestCase):
    def test_chunks_overlap(self):
        text = " ".join(str(i) for i in range(100))
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])

    def test_empty_text(self):
        self.assertEqual(chunk_text("   "), [])

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_size=5, overlap=5)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = RAGPipeline()
        cls.rag.ingest(DOCS)

    def test_retrieves_correct_source(self):
        top = self.rag.retrieve("What are the AQI categories in India?")[0][0]
        self.assertEqual(top["source"], "air_quality.txt")
        top = self.rag.retrieve("What is a plug-in hybrid vehicle?")[0][0]
        self.assertEqual(top["source"], "electric_vehicles.txt")

    def test_ask_returns_answer_and_sources(self):
        result = self.rag.ask("What is chunk overlap used for?")
        self.assertTrue(result["answer"])
        self.assertTrue(result["sources"])

    def test_unrelated_question_is_refused(self):
        result = self.rag.ask("zzzz qqqq xxxx")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
