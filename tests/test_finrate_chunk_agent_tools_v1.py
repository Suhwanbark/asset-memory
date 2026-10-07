"""Check source access and trace-critical offsets, not answer quality."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "method"))
import bm25s
import finrate_chunk_agent_tools_v1 as m


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = [
            {"_id": "doc_000001", "title": "AAA_2024-01-01_10K", "text": "fill " * 600 + "tailanchor revenue 42.7 then revenue 81"},
            {"_id": "doc_000002", "title": "BBB_2023-01-01_10K", "text": "Revenue 20.1. " + "한글 😀 token\n" * 10000},
        ]
        model = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        model.index([m.bm25.tokenize(d["text"]) for d in cls.docs], show_progress=False)
        cls.env = m.ChunkTools(cls.docs, model)

    def test_tail_search_and_exact_excerpts(self):
        result = self.env.call("search", {"query": "tailanchor"})
        self.assertEqual(result["results"][0]["doc_id"], "doc_000001")
        for span in result["results"][0]["excerpts"]:
            self.assertEqual(span["text"], self.docs[0]["text"][span["start"]:span["end"]])

    def test_grep_pagination_and_literal_decimal(self):
        result = self.env.grep("revenue", limit=1)
        self.assertEqual(result["next_offset"], 1)
        second = self.env.grep("revenue", limit=1, offset=1)
        self.assertGreater(second["matches"][0]["match_start"], result["matches"][0]["match_start"])
        self.assertEqual(len(self.env.grep("42.7")["matches"]), 1)
        self.assertEqual(self.env.grep("42x7")["matches"], [])

    def test_title_scope_and_regex(self):
        result = self.env.grep(r"BBB_202[0-9]", regex=True)
        self.assertEqual(result["matches"][0]["field"], "title")
        self.assertEqual(self.env.grep("revenue", doc_ids=[])["matches"], [])

    def test_full_unicode_body_can_be_read_in_pages(self):
        start, pieces = 0, []
        while start is not None:
            result = self.env.call("read", {"doc_id": "doc_000002", "start": start, "length": 1000000})
            self.assertLessEqual(m.tokens(result["text"]), m.READ_TOKENS)
            self.assertEqual(result["text"], self.docs[1]["text"][result["start"]:result["end"]])
            self.assertGreater(result["end"], start)
            pieces.append(result["text"])
            start = result["next_start"]
        self.assertEqual("".join(pieces), self.docs[1]["text"])

    def test_invalid_request_does_not_silently_change_scope(self):
        with self.assertRaises(ValueError):
            self.env.grep("revenue", doc_ids=["doc_missing"])
        with self.assertRaises(ValueError):
            self.env.read("doc_000001", start=-1)
        self.assertEqual(self.env.search("unknownword")["results"], [])


if __name__ == "__main__":
    unittest.main()
