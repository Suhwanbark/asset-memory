"""Offline source/gold/inference-boundary audit; never call a model or the network.

Run from replicate: .venv/bin/python -m unittest discover -s tests -p 'temporal_evidence_test.py' -v
These checks establish integrity and task scope, not causal information utility.
"""
from __future__ import annotations

import copy
from collections import Counter
from datetime import date, datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/temporal_evidence_poc_v1"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "method" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flat(text):
    return re.sub(r"\s+", " ", text).replace("‑", "-").replace("–", "-").strip()


def number(value):
    value = value.strip().replace("‑", "-").replace("–", "-")
    pieces = re.split(r"[- ]", value)
    return sum((Fraction(piece) for piece in pieces if piece), Fraction(0))


def independent_statement_range(text):
    """Read the decision sentence independently of production parse_range()."""
    normalized = flat(text)
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    candidates = [s for s in sentences if "Committee decided to" in s
                  and "target range for the federal funds rate" in s]
    if len(candidates) != 1:
        raise AssertionError(f"Expected one decision sentence, got {candidates!r}")
    rates = re.findall(r"(?:at|to) (\d+(?:[ -]\d+/\d+)?|\d+/\d+) to "
                       r"(\d+(?:[ -]\d+/\d+)?|\d+/\d+) percent", candidates[0])
    if len(rates) != 1:
        raise AssertionError(f"Unambiguous range missing in {candidates[0]!r}")
    return tuple(number(v) for v in rates[0])


def complete_response(case):
    return {"judgments": [{"id": p["id"], "label": "supported", "reason": "Mock only.",
                           "source_ids": []} for p in case["probes"]]}


class TemporalEvidenceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((DATA / "fed_cases.json").read_text())["cases"]
        labels = json.loads((DATA / "fed_gold.json").read_text())["labels"]
        cls.gold = {(g["case_id"], g["probe_id"]): g for g in labels}
        cls.sources = load_module("temporal_sources_under_test", "temporal_evidence_sources.py")
        cls.runner = load_module("temporal_eval_under_test", "temporal_evidence_eval.py")
        cls.calendar = (DATA / "fed_calendar.html").read_text()
        cls.calendar_rows = cls.sources.calendar_rows(cls.calendar)

    def test_fixed_event_count_and_gold_bijection(self):
        self.assertEqual(len(self.cases), 24)
        self.assertEqual(len({c["id"] for c in self.cases}), 24)
        self.assertEqual(Counter(c["event_date"][:4] for c in self.cases),
                         {"2023": 8, "2024": 8, "2025": 8})
        actual = {(c["id"], p["id"]) for c in self.cases for p in c["probes"]}
        self.assertEqual(actual, set(self.gold))
        self.assertEqual(len(actual), 96)
        for case in self.cases:
            self.assertEqual(len(case["probes"]), 4)
            self.assertEqual({p["id"] for p in case["probes"]}, {"p1", "p2", "p3", "p4"})

    def test_official_events_are_not_mislabeled_as_traded_markets(self):
        for case in self.cases:
            self.assertEqual(case["source_stratum"], "official_event_not_market_linked")
            self.assertIsNone(case["market_reference"])
            self.assertIn("overlapping", case["dependence"])

    def test_hashes_publication_dates_and_no_future_document(self):
        rows = {r["stamp"]: r for r in self.calendar_rows}
        seen = set()
        for case in self.cases:
            cutoff = datetime.fromisoformat(case["cutoff"].replace("Z", "+00:00"))
            self.assertEqual(case["cutoff"][-9:], "23:59:59Z")
            current_stamp = case["event_date"].replace("-", "")
            self.assertEqual(cutoff.date().isoformat(), rows[current_stamp]["minutes_date"])
            self.assertEqual(Counter(d["kind"] for d in case["documents"]),
                             {"statement": 4, "minutes": 1})
            for doc in case["documents"]:
                self.assertLessEqual(date.fromisoformat(doc["publication_date"]), cutoff.date())
                self.assertTrue(doc["url"].startswith("https://www.federalreserve.gov/"))
                stamp = re.search(r"fed_(\d{8})_", doc["id"]).group(1)
                expected = rows[stamp]["minutes_date" if doc["kind"] == "minutes" else "date"]
                self.assertEqual(doc["publication_date"], expected)
                if doc["id"] in seen:
                    continue
                seen.add(doc["id"])
                path = Path(doc["text_path"])
                self.assertTrue(path.resolve().is_relative_to(DATA.resolve()))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), doc["text_sha256"])
                raw = path.with_name("source.html").read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), doc["html_sha256"])
                self.assertEqual(len(path.read_text()), doc["characters"])
                self.assertEqual(doc["http_status"], 200)
                soup = BeautifulSoup(raw, "html.parser")
                if doc["kind"] == "statement":
                    published = datetime.strptime(soup.select_one(".article__time").get_text(strip=True),
                                                  "%B %d, %Y").date().isoformat()
                    self.assertEqual(published, doc["publication_date"])
                # Reject silently revised online minutes in this fixed release-time dataset.
                updated = soup.select_one("#lastUpdate").get_text(" ", strip=True)
                updated = updated.split("Last Update:", 1)[1].strip()
                self.assertEqual(datetime.strptime(updated, "%B %d, %Y").date().isoformat(),
                                 doc["publication_date"])

    def test_minutes_are_available_after_meeting_but_before_next_meeting(self):
        stamps = sorted(set(re.findall(r"/monetary(20\d{6})a\.htm", self.calendar)))
        for case in self.cases:
            event = case["event_date"].replace("-", "")
            following = [s for s in stamps if s > event]
            self.assertTrue(following)
            next_date = datetime.strptime(following[0], "%Y%m%d").date()
            cutoff = date.fromisoformat(case["cutoff"][:10])
            self.assertLess(date.fromisoformat(case["event_date"]), cutoff)
            self.assertLess(cutoff, next_date)

    def test_range_parser_numerical_formats_and_actual_decisions(self):
        examples = [
            ("target range for the federal funds rate at 5-1/4 to 5-1/2 percent", (Fraction(21, 4), Fraction(11, 2))),
            ("target range for the federal funds rate by 1/4 percentage point to 3-1/2 to 3‑3/4 percent", (Fraction(7, 2), Fraction(15, 4))),
            ("target range for the federal funds rate to 1/4 to 1/2 percent", (Fraction(1, 4), Fraction(1, 2))),
            ("target range for the federal funds rate at 4.25 to 4.5 percent", (Fraction(17, 4), Fraction(9, 2))),
        ]
        for text, expected in examples:
            parsed = self.sources.parse_range(text)
            self.assertEqual(tuple(number(v) for v in parsed[:2]), expected)
        with self.assertRaises(ValueError):
            self.sources.parse_range("There is no target range in this source.")
        for case in self.cases:
            for doc in case["documents"]:
                if doc["kind"] != "statement":
                    continue
                text = Path(doc["text_path"]).read_text()
                parsed = self.sources.parse_range(text)
                numeric = tuple(number(v) for v in parsed[:2])
                self.assertEqual(numeric, independent_statement_range(text))
                self.assertEqual(numeric[1] - numeric[0], Fraction(1, 4))

    def test_current_gold_uses_actual_current_not_prior_range(self):
        observed = Counter()
        for case in self.cases:
            statements = sorted((d for d in case["documents"] if d["kind"] == "statement"),
                                key=lambda d: d["publication_date"])
            previous, current = statements[-2:]
            old = independent_statement_range(Path(previous["text_path"]).read_text())
            now = independent_statement_range(Path(current["text_path"]).read_text())
            self.assertEqual(current["publication_date"], case["event_date"])
            for probe in case["probes"]:
                gold = self.gold[(case["id"], probe["id"])]
                if gold["category"] != "policy_state":
                    continue
                self.assertEqual(probe["scope"], "current")
                expected = "supported" if old == now else "contradicted"
                self.assertEqual(gold["label"], expected)
                self.assertIn(current["id"], gold["source_ids"])
                self.assertTrue(probe["claim"].startswith("The current federal funds target range is "))
                observed[expected] += 1
        self.assertGreater(observed["supported"], 0)
        self.assertGreater(observed["contradicted"], 0)

    def test_historical_truth_is_not_overwritten_by_current_change(self):
        for case in self.cases:
            docs = {d["id"]: d for d in case["documents"]}
            for probe in case["probes"]:
                gold = self.gold[(case["id"], probe["id"])]
                if gold["category"] != "historical_record":
                    continue
                self.assertEqual(probe["scope"], "historical")
                self.assertEqual(gold["label"], "supported")
                self.assertEqual(len(gold["source_ids"]), 1)
                source = docs[gold["source_ids"][0]]
                self.assertIn(source["publication_date"], probe["claim"])
                bounds = self.sources.parse_range(Path(source["text_path"]).read_text())[:2]
                self.assertIn(f"{bounds[0]} to {bounds[1]} percent", probe["claim"])
                self.assertLess(source["publication_date"], case["event_date"])

    def test_two_percent_objective_is_explicitly_retained(self):
        for case in self.cases:
            source = next(d for d in case["documents"] if d["kind"] == "statement"
                          and d["publication_date"] == case["event_date"])
            self.assertIn("inflation at the rate of 2 percent over the longer run",
                          flat(Path(source["text_path"]).read_text()))
            g = next(self.gold[(case["id"], p["id"])] for p in case["probes"]
                     if self.gold[(case["id"], p["id"])]["category"] == "persistent_objective")
            self.assertEqual(g["label"], "supported")
            self.assertIn(source["id"], g["source_ids"])

    def test_future_outcome_is_not_used_as_present_evidence(self):
        for case in self.cases:
            for probe in case["probes"]:
                g = self.gold[(case["id"], probe["id"])]
                if g["category"] == "unconfirmed_future_commitment":
                    self.assertEqual(g["label"], "insufficient")
                    self.assertIn("next scheduled FOMC meeting", probe["claim"])
                    self.assertIn("do not establish", g["explanation"])
                    self.assertTrue(set(g["source_ids"]) <= {d["id"] for d in case["documents"]})

    def test_all_citations_belong_to_case_and_categories_do_not_enter_probes(self):
        for case in self.cases:
            allowed = {d["id"] for d in case["documents"]}
            for probe in case["probes"]:
                g = self.gold[(case["id"], probe["id"])]
                self.assertTrue(set(g["source_ids"]) <= allowed)
                self.assertEqual(set(probe), {"id", "scope", "claim"})
                self.assertNotIn(g["category"], json.dumps(probe))

    def test_75_percent_of_probes_have_fixed_category_labels_report_limitation(self):
        # This passes to document a limitation, not to certify a hard benchmark.
        fixed = [g for g in self.gold.values() if g["category"] != "policy_state"]
        self.assertEqual(len(fixed), 72)
        for category, label in [("historical_record", "supported"),
                                ("persistent_objective", "supported"),
                                ("unconfirmed_future_commitment", "insufficient")]:
            self.assertEqual({g["label"] for g in fixed if g["category"] == category}, {label})

    def test_packet_rejects_future_metadata_and_changed_source(self):
        original = self.cases[0]
        case = copy.deepcopy(original)
        case["documents"][0]["publication_date"] = "2099-01-01"
        with self.assertRaises(AssertionError):
            self.runner.source_packet(case)
        case = copy.deepcopy(original)
        case["documents"][0]["text_sha256"] = "bad_hash"
        with self.assertRaises(AssertionError):
            self.runner.source_packet(case)

    def test_packet_preserves_complete_sources(self):
        for case in self.cases:
            packet = self.runner.source_packet(case)
            for source in case["documents"]:
                self.assertIn(Path(source["text_path"]).read_text(), packet)
                self.assertIn(source["publication_date"], packet)
                self.assertIn(source["url"], packet)

    def test_memory_formation_does_not_receive_question_or_gold(self):
        case = copy.deepcopy(self.cases[0])
        case["gold"] = "GOLD_CANARY_DO_NOT_EXPOSE"
        case["probes"] = [{"id": "probe_canary", "scope": "current", "claim": "QUERY_CANARY_DO_NOT_EXPOSE"}]
        captured = []
        with patch.object(self.runner, "invoke", side_effect=lambda prompt, folder:
                          captured.append(prompt) or {"text": "Mock memory"}):
            self.assertEqual(self.runner.form_summary(case, "OFFICIAL_SOURCE_CANARY"), "Mock memory")
        self.assertEqual(len(captured), 1)
        self.assertIn("OFFICIAL_SOURCE_CANARY", captured[0])
        self.assertNotIn("GOLD_CANARY", captured[0])
        self.assertNotIn("QUERY_CANARY", captured[0])
        self.assertNotIn(case["cutoff"], captured[0])

    def test_final_prompt_respects_scope_without_gold(self):
        case = copy.deepcopy(self.cases[0])
        case["gold"] = "GOLD_CANARY_DO_NOT_EXPOSE"
        prompt = self.runner.build_prompt(case, "RESEARCH_MEMORY_CANARY")
        self.assertIn(case["cutoff"], prompt)
        self.assertIn("historical means what was true or reported", prompt)
        self.assertIn("contradicted if supplied evidence establishes the opposite in the same scope", prompt)
        for p in case["probes"]:
            self.assertIn(p["claim"], prompt)
        self.assertNotIn("GOLD_CANARY", prompt)

    def test_v2_clarifies_latest_reported_state_without_changing_question(self):
        case = self.cases[0]
        with patch.object(self.runner, "INSTRUCTION_VERSION", "v1"):
            original = self.runner.build_prompt(case, "SAME_MEMORY_CANARY")
        with patch.object(self.runner, "INSTRUCTION_VERSION", "v2"):
            clarified = self.runner.build_prompt(case, "SAME_MEMORY_CANARY")
        self.assertNotEqual(original, clarified)
        self.assertIn("latest applicable state established by", clarified)
        self.assertTrue(original.endswith(self.runner.question(case)))
        self.assertTrue(clarified.endswith(self.runner.question(case)))
        self.assertEqual(original.count("SAME_MEMORY_CANARY"), 1)
        self.assertEqual(clarified.count("SAME_MEMORY_CANARY"), 1)

    def test_summary_formation_cache_is_shared_across_instruction_versions(self):
        case = self.cases[0]
        requests = []
        with patch.object(self.runner, "invoke", side_effect=lambda prompt, folder:
                          requests.append((prompt, folder)) or {"text": "SAME_SUMMARY"}):
            for version in ("v1", "v2"):
                with patch.object(self.runner, "INSTRUCTION_VERSION", version):
                    self.runner.form_summary(case, "SAME_OFFICIAL_PACKET")
        self.assertEqual(requests[0], requests[1])
        self.assertTrue(requests[0][1].is_relative_to(self.runner.BASE_OUT))

    def test_rag_query_uses_no_gold_and_chunks_have_source_dates(self):
        case = copy.deepcopy(self.cases[0])
        case["gold"] = "GOLD_CANARY_DO_NOT_EXPOSE"
        text, retrieval = self.runner.rag(case, k=10)
        self.assertEqual(len(retrieval["selected"]), 10)
        self.assertGreater(retrieval["total_chunks"], 10)
        self.assertNotIn("GOLD_CANARY", text)
        sources = {d["id"]: d for d in case["documents"]}
        for chunk in retrieval["selected"]:
            self.assertEqual(chunk["date"], sources[chunk["source_id"]]["publication_date"])
            self.assertLessEqual(chunk["date"], case["cutoff"][:10])
            self.assertGreater(chunk["score"], 0)
            self.assertLessEqual(len(chunk["text"].split()), 220)

    def test_json_validation_rejects_wrong_duplicate_missing_labels(self):
        case = self.cases[0]
        valid = complete_response(case)
        self.runner.validate_response(valid, case)
        for kind in ("duplicate", "missing", "wrong_label", "missing_reason"):
            broken = copy.deepcopy(valid)
            if kind == "duplicate": broken["judgments"][-1] = broken["judgments"][0]
            elif kind == "missing": broken["judgments"].pop()
            elif kind == "wrong_label": broken["judgments"][0]["label"] = "true"
            else: broken["judgments"][0].pop("reason")
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                self.runner.validate_response(broken, case)

    def test_plain_or_fenced_json_but_not_commentary_parses(self):
        obj = complete_response(self.cases[0])
        text = json.dumps(obj)
        self.assertEqual(self.runner.parse_json(text), obj)
        self.assertEqual(self.runner.parse_json("```json\n" + text + "\n```"), obj)
        with self.assertRaises(json.JSONDecodeError):
            self.runner.parse_json("Here is my answer: " + text)

    def test_cache_rejects_changed_prompt_or_model_without_calling_runtime(self):
        with tempfile.TemporaryDirectory(prefix="temporal-cache-test-") as temp:
            folder = Path(temp)
            cached = {"prompt_sha256": self.runner.sha("fixed prompt"),
                      "model": self.runner.MODEL, "text": "cached response"}
            self.runner.save(folder / "response.json", cached)
            with patch.object(self.runner.subprocess, "run", side_effect=AssertionError("No runtime allowed")):
                self.assertEqual(self.runner.invoke("fixed prompt", folder), cached)
                with self.assertRaises(RuntimeError):
                    self.runner.invoke("different prompt", folder)
                cached["model"] = "different model"
                self.runner.save(folder / "response.json", cached)
                with self.assertRaises(RuntimeError):
                    self.runner.invoke("fixed prompt", folder)

    def test_run_case_conditions_share_queries_and_preserve_separate_inputs(self):
        case = self.cases[0]
        requests = []

        def fake_invoke(prompt, folder):
            requests.append((prompt, folder))
            if "Create a concise research memory" in prompt:
                text = "MEMORY_FORMATION_CANARY"
            else:
                text = json.dumps(complete_response(case))
            return {"text": text, "prompt_sha256": self.runner.sha(prompt),
                    "prompt_tokens_o200k": self.runner.tokens(prompt)}

        with tempfile.TemporaryDirectory(prefix="temporal-run-test-") as temp, \
                patch.object(self.runner, "OUT", Path(temp)), \
                patch.object(self.runner, "invoke", side_effect=fake_invoke):
            results = self.runner.run_case(case, ["full_history", "naive_rag_top10", "concise_summary"])
        self.assertEqual(len(results), 3)
        self.assertEqual(len(requests), 4)
        final_prompts = [p for p, _ in requests if "Create a concise research memory" not in p]
        for prompt in final_prompts:
            self.assertTrue(prompt.endswith(self.runner.question(case)))
        self.assertIn("MEMORY_FORMATION_CANARY", final_prompts[-1])
        self.assertGreater(results[0]["delivered_tokens"], results[-1]["delivered_tokens"])


class TemporalEvidenceCleanCohortTests(unittest.TestCase):
    """Audit the fixed source-QC cohort without modifying prior runs or labels."""

    @classmethod
    def setUpClass(cls):
        cls.out = ROOT / "result/temporal_evidence_poc_v3"
        cls.report = json.loads((cls.out / "reviewed_report.json").read_text())
        cls.corrected = json.loads((cls.out / "corrected_case.json").read_text())
        labels = json.loads((cls.out / "reviewed_gold.json").read_text())["labels"]
        cls.gold = {(g["case_id"], g["probe_id"]): g for g in labels}
        cases = []
        for filename in ("fed_cases.json", "corporate_cases.json"):
            cases.extend(json.loads((DATA / filename).read_text())["cases"])
        cls.original_cases = {c["id"]: c for c in cases}
        cls.cases = {cid: cls.original_cases[cid] for cid in cls.report["case_ids"]}
        cls.cases[cls.corrected["id"]] = cls.corrected
        cls.conditions = {"full_history", "naive_rag_top10", "concise_summary"}
        cls.old_id = "corp_irbt_20240129"
        cls.new_id = "corp_irbt_20240129_clean_v3"

    def test_clean_cohort_has_29_events_111_probes_and_no_missing_conditions(self):
        report = self.report
        self.assertEqual(report["event_count"], 29)
        self.assertEqual(report["probe_count"], 111)
        self.assertEqual(len(report["case_ids"]), len(set(report["case_ids"])))
        self.assertEqual(len(self.gold), 111)
        self.assertEqual(len(report["rows"]), 333)
        self.assertEqual(len(report["result_references"]), 87)
        expected = {(cid, condition) for cid in report["case_ids"] for condition in self.conditions}
        actual = {(r["case_id"], r["condition"]) for r in report["result_references"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(report["result_references"]))
        self.assertEqual({g[0] for g in self.gold}, set(report["case_ids"]))
        self.assertNotIn("corporate_vaneck_spot_bitcoin", report["case_ids"])
        self.assertIn("corporate_vaneck_spot_bitcoin", self.original_cases)
        self.assertEqual({e["case_id"] for e in report["exclusions"]},
                         {"corporate_vaneck_spot_bitcoin"})
        self.assertIn("provenance", report["exclusions"][0]["reason"])
        for condition in self.conditions:
            self.assertEqual(report["summary"][condition]["n"], 111)
            self.assertEqual(report["summary"][condition]["case_count"], 29)

    def test_clean_cohort_stale_premise_acceptance_is_zero_of_fifteen_per_condition(self):
        # Recalculate from raw labels; do not just trust the displayed summary.
        for condition in self.conditions:
            rows = [r for r in self.report["rows"] if r["condition"] == condition]
            stale = [r for r in rows if self.gold[(r["case_id"], r["probe_id"])]["label"] == "contradicted"]
            self.assertEqual(len(stale), 15)
            self.assertEqual(sum(r["predicted"] == "supported" for r in stale), 0)
            self.assertEqual(sum(r["predicted"] == "contradicted" for r in stale), 15)
            self.assertEqual(self.report["summary"][condition]["gold_contradicted"], 15)
            self.assertEqual(self.report["summary"][condition]["contradicted->contradicted"], 15)
            policy_changed = [r for r in stale if r["category"] == "policy_state"]
            self.assertEqual(len(policy_changed), 10)
            for row in stale:
                probe = next(p for p in self.cases[row["case_id"]]["probes"] if p["id"] == row["probe_id"])
                self.assertEqual(probe["scope"], "current")

    def test_clean_copy_preserves_original_text_and_identical_raw_html(self):
        old_dir = DATA / "sources" / self.old_id
        new_dir = DATA / "sources" / self.new_id
        old = json.loads((old_dir / "meta.json").read_text())
        new = json.loads((new_dir / "meta.json").read_text())
        old_text = (old_dir / "source.txt").read_bytes()
        new_text = (new_dir / "source.txt").read_bytes()
        old_raw = (old_dir / "source.html").read_bytes()
        new_raw = (new_dir / "source.html").read_bytes()
        self.assertEqual(hashlib.sha256(old_text).hexdigest(), old["text_sha256"])
        self.assertEqual(hashlib.sha256(new_text).hexdigest(), new["text_sha256"])
        self.assertEqual(new["derived_from_text_sha256"], old["text_sha256"])
        self.assertEqual(new["derived_from_source_id"], self.old_id)
        self.assertEqual(old_raw, new_raw)
        self.assertEqual(hashlib.sha256(old_raw).hexdigest(), old["raw_sha256"])
        self.assertEqual(hashlib.sha256(new_raw).hexdigest(), new["raw_sha256"])
        self.assertNotEqual(old_text, new_text)
        self.assertIn(b"Trending news and stories", old_text)
        self.assertNotIn(b"Trending news and stories", new_text)
        self.assertFalse(new["source_snapshot_unchanged_since_publication_verified"])
        original_case = self.original_cases[self.corrected["id"]]
        self.assertIn(self.old_id, {d["id"] for d in original_case["documents"]})
        self.assertNotIn(self.new_id, {d["id"] for d in original_case["documents"]})

    def test_clean_text_excludes_only_identified_dynamic_article_module(self):
        new_dir = DATA / "sources" / self.new_id
        meta = json.loads((new_dir / "meta.json").read_text())
        soup = BeautifulSoup((new_dir / "source.html").read_bytes(), "html.parser")
        article = soup.select_one("article.ArticlePage-mainContent")
        self.assertIsNotNone(article)
        modules = article.select(".articles-links")
        self.assertEqual(len(modules), 1)
        removed = modules[0].get_text(" ", strip=True)
        self.assertIn("Trending news and stories", removed)
        self.assertIn("2026", removed)
        for module in modules:
            module.decompose()
        for node in article.select("script,style,svg,noscript"):
            node.decompose()
        independently_extracted = article.get_text("\n", strip=True).replace("\xa0", " ") + "\n"
        actual = (new_dir / "source.txt").read_text()
        self.assertEqual(actual, independently_extracted)
        self.assertIn("mutual agreement to terminate", actual)
        self.assertIn("Colin Angle", actual)
        self.assertIn("termination fee", actual)
        self.assertNotIn("Prime Big Deal Days 2026", actual)
        self.assertEqual(meta["excluded_module_selectors"], [".articles-links"])
        self.assertTrue(meta["all_article_body_alphanumeric_characters_preserved"])
        self.assertFalse(meta["all_main_document_alphanumeric_characters_preserved"])

    def test_reviewed_result_references_hashes_gold_and_clean_source_ids_align(self):
        rows = {(r["case_id"], r["condition"], r["probe_id"]): r for r in self.report["rows"]}
        self.assertEqual(len(rows), len(self.report["rows"]))
        for reference in self.report["result_references"]:
            path = Path(reference["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), reference["sha256"])
            result = json.loads(path.read_text())
            self.assertEqual(result["case_id"], reference["case_id"])
            self.assertEqual(result["condition"], reference["condition"])
            case = self.cases[result["case_id"]]
            self.assertEqual({j["id"] for j in result["judgments"]}, {p["id"] for p in case["probes"]})
            if result["case_id"] == self.corrected["id"]:
                self.assertTrue(path.is_relative_to(self.out))
                manifest = json.loads((path.parent.parent / "input_manifest.json").read_text())
                self.assertEqual(manifest["case"], self.corrected)
                if result["condition"] in {"full_history", "naive_rag_top10"}:
                    prompt = path.with_name("prompt.txt").read_text()
                    self.assertIn(self.new_id, prompt)
                    self.assertNotIn("Trending news and stories", prompt)
            else:
                self.assertTrue(path.is_relative_to(ROOT / "result/temporal_evidence_poc_v2"))
            for judgment in result["judgments"]:
                key = (result["case_id"], judgment["id"])
                gold = self.gold[key]
                row = rows[(result["case_id"], result["condition"], judgment["id"])]
                self.assertEqual(row["predicted"], judgment["label"])
                self.assertEqual(row["gold"], gold["label"])
                self.assertEqual(row["correct"], judgment["label"] == gold["label"])
                self.assertTrue(set(gold["source_ids"]) <= {d["id"] for d in case["documents"]})
                if key[0] == self.corrected["id"]:
                    self.assertNotIn(self.old_id, gold["source_ids"])
        self.assertTrue(any(self.new_id in g["source_ids"] for g in self.gold.values()))


if __name__ == "__main__":
    unittest.main()
