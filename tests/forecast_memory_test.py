"""Offline integrity/boundary tests; never run a model or change experiment results.

Run: python -m unittest discover -s replicate/tests -p '*_test.py' -v
The annotation checks below exercise rubric semantics, not an automatic LLM judge.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/forecast_memory_smoke_v1"
RUNNER = ROOT / "method/forecast_memory_smoke.py"


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def revenue_relation(forecast, actual):
    """Evaluator-side Decimal comparison; range endpoints count as within range."""
    actual = Decimal(str(actual))
    lo, hi, mid = (Decimal(str(forecast[k])) for k in ("lower", "upper", "midpoint"))
    if actual < lo:
        return "below_prior_guidance_range"
    if actual > hi:
        return "above_prior_guidance_range"
    if actual < mid:
        return "within_prior_guidance_range_below_midpoint"
    if actual > mid:
        return "within_prior_guidance_range_above_midpoint"
    return "within_prior_guidance_range_at_midpoint"


def annotation_flags(present, status_correct=None, period_correct=None):
    """Explicit human-annotation logic, not text inference or case scoring."""
    return {
        "omission": not present,
        "temporal_status_error": present and status_correct is False,
        "period_error": present and period_correct is False,
    }


class ForecastMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text())
        cls.cases = cls.manifest["cases"]
        cls.gold = {c["id"]: c for c in json.loads((DATA / "gold.json").read_text())["cases"]}
        spec = importlib.util.spec_from_file_location("forecast_smoke_under_test", RUNNER)
        cls.runner = importlib.util.module_from_spec(spec)
        # Importing the runner adjusts SDK env vars; restore the caller environment.
        with patch.dict(os.environ):
            spec.loader.exec_module(cls.runner)

    def test_case_identity_and_gold_alignment(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.gold))
        self.assertEqual(len({c["company"] for c in self.cases}), len(ids))
        self.assertEqual(len({c["past"]["url"] for c in self.cases}), len(ids))

    def test_chronology_and_source_dates(self):
        for c in self.cases:
            with self.subTest(case=c["id"]):
                self.assertLess(date.fromisoformat(c["past_date"]), date.fromisoformat(c["current_date"]))
                for phase in ("past", "current"):
                    self.assertEqual(c[f"{phase}_date"], c[phase]["publication_date"])
                    self.assertEqual(c[f"{phase}_date"], self.gold[c["id"]][f"{phase}_date"])
                    self.assertEqual(c["company"], c[phase]["publisher"])

    def test_preserved_source_hashes_sizes_and_tables(self):
        for c in self.cases:
            for phase in ("past", "current"):
                with self.subTest(case=c["id"], phase=phase):
                    source = c[phase]
                    for kind in ("html", "text"):
                        path = Path(source[f"{kind}_path"])
                        self.assertTrue(path.resolve().is_relative_to(DATA.resolve()))
                        raw = path.read_bytes()
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), source[f"{kind}_sha256"])
                        if kind == "html":
                            self.assertEqual(len(raw), source["html_bytes"])
                    text = Path(source["text_path"]).read_text()
                    self.assertEqual(len(text), source["text_characters"])
                    self.assertGreater(len(text), 10000)
                    self.assertGreater(source["table_count"], 0)
                    self.assertEqual(text.count("[TABLE]"), source["table_count"])
                    self.assertEqual(text.count("[/TABLE]"), source["table_count"])
                    self.assertEqual(source["http_status"], 200)

    def test_gold_full_excerpts_are_in_archived_sources(self):
        for c in self.cases:
            for phase, field in (("past", "forecast"), ("current", "reported_actual")):
                with self.subTest(case=c["id"], phase=phase):
                    gold_fact = self.gold[c["id"]][field]
                    self.assertEqual(gold_fact["source_url"], c[phase]["url"])
                    self.assertIn(normalized(gold_fact["source_excerpt"]),
                                  normalized(Path(c[phase]["text_path"]).read_text()))

    def test_gold_range_construction_and_relations(self):
        for case_id, gold in self.gold.items():
            with self.subTest(case=case_id):
                f = gold["forecast"]
                mid = Decimal(str(f["midpoint"]))
                lo, hi = Decimal(str(f["lower"])), Decimal(str(f["upper"]))
                self.assertLessEqual(lo, mid)
                self.assertLessEqual(mid, hi)
                if "relative_tolerance" in f:
                    delta = mid * Decimal(str(f["relative_tolerance"]))
                    self.assertEqual((lo, hi), (mid - delta, mid + delta))
                if "absolute_tolerance" in f:
                    delta = Decimal(str(f["absolute_tolerance"]))
                    self.assertEqual((lo, hi), (mid - delta, mid + delta))
                self.assertEqual(revenue_relation(f, gold["reported_actual"]["value"]), gold["comparison"])
                self.assertEqual(revenue_relation(f, gold["reported_actual"]["release_rounded_value"]),
                                 gold["comparison"])

    def test_intel_below_midpoint_is_not_below_guidance(self):
        gold = self.gold["intel_2024_q2"]
        self.assertEqual(revenue_relation(gold["forecast"], 12.833),
                         "within_prior_guidance_range_below_midpoint")
        self.assertEqual(revenue_relation(gold["forecast"], 12.5),
                         "within_prior_guidance_range_below_midpoint")
        self.assertEqual(revenue_relation(gold["forecast"], 13.5),
                         "within_prior_guidance_range_above_midpoint")
        self.assertEqual(revenue_relation(gold["forecast"], 12.499), "below_prior_guidance_range")

    def test_source_prompt_and_history_only_read_complete_past_release(self):
        runner = self.runner
        for original in self.cases:
            with self.subTest(case=original["id"]):
                case = dict(original)
                case["current"] = {"text_path": "/forbidden/current.txt", "secret": "FUTURE_CANARY"}
                case["gold"] = "GOLD_CANARY"
                expected_path = Path(case["past"]["text_path"])
                source_text = expected_path.read_text()
                reads, requests = [], []

                def guarded_read(path, *args, **kwargs):
                    reads.append(path)
                    self.assertEqual(path, expected_path, "Source builder attempted non-past file access")
                    return source_text

                def fake_generate(prompt, folder, **kwargs):
                    requests.append((prompt, kwargs))
                    return {"text": "Research note from past evidence.", "tool_calls": [{"command": "read source.txt"}]}

                with patch.object(Path, "read_text", guarded_read), patch.object(runner, "generate", fake_generate), \
                        patch.object(runner, "save"):
                    history = runner.source(case)
                self.assertTrue(reads)
                prompt, kwargs = requests[0]
                self.assertIn(source_text, kwargs["source_text"])
                self.assertIn(source_text, history[1]["content"])
                combined = prompt + kwargs["source_text"] + json.dumps(history)
                self.assertNotIn("FUTURE_CANARY", combined)
                self.assertNotIn("GOLD_CANARY", combined)
                self.assertNotIn(case["current_date"], prompt)
                self.assertIn("Do not read any other file", prompt)
                self.assertIn("Do not read any other file or use the network", prompt)

    def test_native_memory_add_receives_no_future_question_or_gold(self):
        case = self.cases[0]
        calls = []

        class FakeMemory:
            def __init__(self):
                self.vector_store = types.SimpleNamespace(client=types.SimpleNamespace(close=lambda: None))

            @classmethod
            def from_config(cls, config):
                return cls()

            def add(self, messages, **kwargs):
                calls.append(("add", messages, kwargs))
                return {"results": []}

            def get_all(self, **kwargs):
                calls.append(("get_all", kwargs))
                return {"results": []}

            def search(self, query, **kwargs):
                calls.append(("search", query, kwargs))
                return {"results": []}

        fake_module = types.ModuleType("mem0")
        fake_module.Memory = FakeMemory
        history = [{"role": "assistant", "content": "PAST_ONLY_CANARY"}]
        with tempfile.TemporaryDirectory(prefix="forecast-memory-test-") as tmp, \
                patch.dict(sys.modules, {"mem0": fake_module}), \
                patch.object(self.runner, "OUT", Path(tmp)), patch.object(self.runner, "save"):
            self.runner.memory(case, history, object())
        self.assertEqual([c[0] for c in calls], ["add", "get_all", "search"])
        self.assertEqual(calls[0][1], history)
        self.assertNotIn(case["current_date"], json.dumps(calls[0]))
        self.assertEqual(calls[0][2]["metadata"]["publication_date"], case["past_date"])
        self.assertEqual(calls[2][1], self.runner.question(case))

    def test_runner_does_not_load_gold(self):
        tree = ast.parse(RUNNER.read_text())
        # Data labels may be described in documentation, but no gold-path literal
        # or gold-key lookup belongs to the inference implementation.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotEqual(node.value, "gold.json")
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                self.assertNotIn(node.slice.value, ("gold", "reported_actual", "comparison", "forecast"))

    def test_omission_is_not_temporal_status_error(self):
        self.assertEqual(annotation_flags(False),
                         {"omission": True, "temporal_status_error": False, "period_error": False})
        self.assertEqual(annotation_flags(True, status_correct=False, period_correct=True),
                         {"omission": False, "temporal_status_error": True, "period_error": False})
        self.assertEqual(annotation_flags(True, status_correct=True, period_correct=False),
                         {"omission": False, "temporal_status_error": False, "period_error": True})
        self.assertEqual(annotation_flags(True, status_correct=None, period_correct=None),
                         {"omission": False, "temporal_status_error": False, "period_error": False})


if __name__ == "__main__":
    unittest.main()
