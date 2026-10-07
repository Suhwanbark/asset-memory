"""Meaningful integrity checks; synthetic fixtures are not benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "method"))
from finrate_poc.bank import Bank, source_span
from finrate_poc.data import metadata
from finrate_poc.runtime import DiskLow, check_disk
from finrate_poc.cohort import approve_gate, freeze
from finrate_poc.data import MODEL, MODEL_CUTOFF


class BankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "bank.sqlite"
        self.bank = Bank(self.path)
        self.s1 = {"id": "s1", "published": "2025-02-01", "title": "A 2024 annual report",
                   "text": "In 2024 we relied on a single external manufacturer for product X."}
        self.s2 = {"id": "s2", "published": "2025-05-01", "title": "A 2025 Q1 report",
                   "text": "In Q1 2025 we qualified a second manufacturer for product X."}
        self.bank.register_source(self.s1)
        self.f1 = self.bank.add_fact(self.fact(self.s1, "single manufacturer", "2024"))
        self.bank.register_source(self.s2)
        self.f2 = self.bank.add_fact(self.fact(self.s2, "second manufacturer qualified", "2025-Q1"))

    def fact(self, source, value, period):
        return {"subject": "A", "topic": "product X manufacturing", "predicate": "manufacturing arrangement",
                "value": value, "period": period, "modality": "reported", "source_id": source["id"], "quote": source["text"]}

    def tearDown(self):
        self.bank.close()
        self.tmp.cleanup()

    def test_history_keeps_old_observation_and_separates_dates(self):
        self.bank.link(old_id=self.f1, new_id=self.f2, relation="updates",
                       rationale="A second manufacturer is newly qualified", evidence_quote=self.s2["text"])
        history = self.bank.history(self.f2)
        self.assertEqual(len(history["facts"]), 2)
        old = self.bank.read_fact(self.f1)
        self.assertEqual(old["period"], "2024")
        self.assertEqual(old["observed_at"], "2025-02-01")
        self.assertEqual(old["value"], "single manufacturer")

    def test_fabricated_quote_is_rejected(self):
        bad = self.fact(self.s2, "invented supplier", "2025-Q1")
        bad["quote"] = "We switched all manufacturing to another company."
        with self.assertRaises(ValueError):
            self.bank.add_fact(bad)

    def test_whitespace_alignment_preserves_original_span(self):
        text = "Authorized 9.5\u00a0million\n shares for repurchase."
        proposed = "Authorized 9.5 million shares for repurchase."
        start, end, mode = source_span(text, proposed)
        self.assertEqual(text[start:end], text)
        self.assertEqual(mode, "whitespace_normalized")

    def test_quote_alignment_never_changes_a_number(self):
        with self.assertRaises(ValueError):
            source_span("Authorized 9.5 million shares for repurchase.", "Authorized 9.6 million shares for repurchase.")

    def test_no_implicit_retraction(self):
        with self.assertRaises(ValueError):
            self.bank.link(old_id=self.f1, new_id=self.f2, relation="retracts",
                           rationale="Old supplier absent", evidence_quote=self.s2["text"])

    def test_backward_links_are_rejected(self):
        with self.assertRaises(ValueError):
            self.bank.link(old_id=self.f2, new_id=self.f1, relation="updates",
                           rationale="Backwards", evidence_quote=self.s1["text"])

    def test_source_overwrite_is_rejected(self):
        with self.assertRaises(ValueError):
            self.bank.register_source({**self.s2, "text": "Changed source"})

    def test_asof_grep_excludes_later_observation(self):
        hits = self.bank.grep("manufacturer", before="2025-03-01")
        self.assertEqual([h["id"] for h in hits], [self.f1])

    def test_readonly_evaluation_cannot_modify_bank(self):
        frozen = Bank(self.path, readonly=True)
        try:
            with self.assertRaises(PermissionError):
                frozen.tool("add_fact", {"fact": self.fact(self.s2, "bad", None)})
            self.assertEqual(frozen.read_fact(self.f1)["source_id"], "s1")
        finally:
            frozen.close()

    def test_event_chain_detects_tampering(self):
        self.bank.verify_event_chain()
        self.bank.db.execute("UPDATE events SET body='{}' WHERE seq=1")
        with self.assertRaises(ValueError):
            self.bank.verify_event_chain()

    def test_checkpoint_cannot_be_replaced(self):
        self.bank.checkpoint("2025-Q2")
        self.bank.add_fact(self.fact(self.s2, "another observation", "2025-Q1"))
        with self.assertRaises(ValueError):
            self.bank.checkpoint("2025-Q2")

    def test_grep_is_literal(self):
        self.assertEqual(self.bank.grep(".*"), [])


class GuardTests(unittest.TestCase):
    def test_empty_post_cutoff_set_cannot_start_inference(self):
        with self.assertRaisesRegex(ValueError, "No post-cutoff"):
            approve_gate({"knowledge_cutoff": MODEL_CUTOFF, "date_candidates": 0}, {"model": MODEL})

    def test_cutoff_cannot_be_silently_relaxed(self):
        with self.assertRaises(ValueError):
            approve_gate({}, {"model": MODEL, "cutoff_policy": "exploratory_exposure_uncontrolled"})

    def test_low_disk_stops_execution(self):
        from collections import namedtuple
        usage = namedtuple("Usage", "total used free")
        with patch("finrate_poc.runtime.shutil.disk_usage", return_value=usage(20, 15, 5)):
            with self.assertRaises(DiskLow):
                check_disk(minimum=10)

    def test_metadata_does_not_take_dates_from_body(self):
        m = metadata({"_id": "d", "title": "A_2025-06-09_10-K", "text": "Plan for 2030-01-01"})
        self.assertEqual(m["filing_date"], "2025-06-09")

    def test_ambiguous_title_date_is_unknown(self):
        m = metadata({"_id": "d", "title": "A_2024-01-01_2025-01-01", "text": ""})
        self.assertIsNone(m["filing_date"])


class CohortTests(unittest.TestCase):
    def test_split_keeps_overlapping_sources_together_and_omits_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "qa").mkdir()
            metadata_rows = [{"doc_id": f"d{i}", "ticker": "A", "filing_date": "2025-01-01"} for i in range(6)]
            questions = [{"q_id": f"q{i}", "question": f"Question {i}", "answer": "secret",
                          "key_points": ["secret"], "doc_ids": [f"d{i//2}"]} for i in range(12)]
            (base / "qa/LT-QA.json").write_text(json.dumps(questions))
            policy = {"model": MODEL, "cutoff_policy": "exploratory_exposure_uncontrolled",
                      "user_authorization": "unit-test fixture only", "asset_count": 1,
                      "dev_count": 20, "eval_count": 20}
            frozen = freeze(base, base / "out", {}, policy, source_metadata=metadata_rows)
            dev = json.loads((base / "out/gold/dev.json").read_text())
            test = json.loads((base / "out/gold/evaluation.json").read_text())
            self.assertFalse({s for q in dev for s in q["doc_ids"]} & {s for q in test for s in q["doc_ids"]})
            inputs = json.loads((base / "out/dev_inputs.json").read_text())
            self.assertTrue(all(set(q) == {"q_id", "question", "task"} for q in inputs))
            self.assertTrue(frozen["no_model_scores_used"])

    def test_recent_filing_date_alone_cannot_approve_a_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "qa").mkdir()
            (base / "qa/LT-QA.json").write_text(json.dumps([
                {"q_id": "q", "question": "Old fact repeated?", "answer": "yes", "key_points": ["old"], "doc_ids": ["d"]}]))
            with self.assertRaisesRegex(ValueError, "No approved eligible"):
                freeze(base, base / "out", {"knowledge_cutoff": MODEL_CUTOFF, "date_candidates": 1},
                       {"model": MODEL}, source_metadata=[{"doc_id": "d", "ticker": "A", "filing_date": "2026-08-01"}])

    def test_asset_memory_is_replayable_and_period_checkpoints_exist(self):
        from finrate_poc.memory import build_asset_memory

        class FixtureModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt, folder, **kwargs):
                self.calls += 1
                if "candidate history" in prompt:
                    return '{"links":[]}'
                obj = json.loads(prompt.rsplit("\n", 1)[1])
                return json.dumps({"facts": [{"subject": obj["asset"], "topic": "manufacturing",
                     "predicate": "arrangement", "value": "source observation", "period": None,
                     "modality": "reported", "source_id": obj["source_id"], "quote": obj["window"]}]})

        docs = [{"_id": "s1", "title": "A_2025-02-01_10-K", "text": "We use one external manufacturer."},
                {"_id": "s2", "title": "A_2025-05-01_10-Q", "text": "We qualified a second external manufacturer."}]
        with tempfile.TemporaryDirectory() as tmp:
            model = FixtureModel()
            build_asset_memory(docs, tmp, model)
            calls = model.calls
            build_asset_memory(docs, tmp, model)
            self.assertEqual(model.calls, calls)
            bank = Bank(Path(tmp) / "v1/A/bank.sqlite", readonly=True)
            try:
                labels = [r[0] for r in bank.db.execute("SELECT label FROM checkpoints ORDER BY label")]
                self.assertEqual(labels, ["2025-Q1", "2025-Q2"])
                self.assertEqual(len(bank.grep("manufacturer")), 2)
            finally:
                bank.close()


class PredictionTests(unittest.TestCase):
    def test_naive_condition_never_reads_index_or_memory(self):
        from finrate_poc.agent import predict

        class NoIndex:
            def search(self, *args, **kwargs):
                raise AssertionError("Naive LLM touched retrieval")

        class FixtureModel:
            def invoke(self, prompt, folder):
                return '{"answer":"Unknown", "claims":[]}'

        with tempfile.TemporaryDirectory() as tmp:
            result = predict("What changed?", "naive_llm", NoIndex(), [], {}, FixtureModel(), tmp)
            self.assertEqual(result["evidence"], [])
            self.assertEqual(result["model_calls"], 1)

    def test_agent_uses_memory_tools_without_mutating_shared_bank(self):
        from finrate_poc.agent import predict
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.sqlite"
            bank = Bank(path)
            bank.register_source({"id": "d", "published": "2025-01-01", "title": "A_2025-01-01_10-K",
                                  "text": "We use an external manufacturer."})
            fid = bank.add_fact({"subject": "A", "topic": "manufacturing", "predicate": "relies on",
                 "value": "external manufacturer", "period": None, "modality": "reported", "source_id": "d",
                 "quote": "We use an external manufacturer."})
            before = bank.verify_event_chain()
            bank.close()
            frozen = Bank(path, readonly=True)

            class EmptyIndex:
                def search(self, *args, **kwargs):
                    return []

            class FixtureModel:
                def __init__(self):
                    self.turn = 0

                def invoke(self, prompt, folder):
                    self.turn += 1
                    if self.turn == 1:
                        return '{"reason":"find history", "tool":"memory_grep", "arguments":{"asset":"A","pattern":"manufacturer"}}'
                    if self.turn == 2:
                        return '{"reason":"enough", "tool":"finish", "arguments":{}}'
                    return json.dumps({"answer": "Uses external manufacturing", "claims": [{"text": "External manufacturing", "evidence_ids": [fid]}]})

            try:
                result = predict("Does A use external manufacturing?", "agentic", EmptyIndex(), [],
                                 {"A": frozen}, FixtureModel(), Path(tmp) / "prediction")
                self.assertEqual(result["model_calls"], 3)
                self.assertEqual(result["invalid_citation_ids"], [])
                self.assertEqual(frozen.verify_event_chain(), before)
                self.assertEqual(result["evidence"][0]["id"], fid)
            finally:
                frozen.close()

    def test_nonexistent_citations_are_recorded(self):
        from finrate_poc.agent import predict

        class FixtureModel:
            def invoke(self, prompt, folder):
                return '{"answer":"Unsupported", "claims":[{"text":"invented", "evidence_ids":["absent"]}]}'

        with tempfile.TemporaryDirectory() as tmp:
            result = predict("Q", "naive_llm", None, [], {}, FixtureModel(), tmp)
            self.assertEqual(result["invalid_citation_ids"], ["absent"])


if __name__ == "__main__":
    unittest.main()
