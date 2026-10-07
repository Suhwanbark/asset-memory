"""Aggregate a completed smoke with explicitly reviewed, non-automatic claim labels."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/forecast_memory_smoke_v1"


def main():
    # Manual semantic review of source/store/retrieval/output, NOT an accuracy judge.
    coverage = {
        "nvidia_2022_q2": {"full_history": "explicit_numeric_comparison", "no_memory": "justified_abstention",
                           "mem0_top10": "comparison_omitted", "mem0_all": "comparison_omitted"},
        "amd_2022_q3": {"full_history": "explicit_numeric_comparison", "no_memory": "justified_abstention",
                        "mem0_top10": "explicit_numeric_comparison", "mem0_all": "explicit_numeric_comparison"},
        "intel_2024_q2": {"full_history": "comparison_omitted", "no_memory": "justified_abstention",
                          "mem0_top10": "comparison_omitted", "mem0_all": "correct_qualitative_comparison"},
    }
    rows = []
    for case_id, conditions in coverage.items():
        bundle = json.loads((OUT/case_id/"mem0/bundle.json").read_text())
        all_ids = {m["id"] for m in bundle["all"]["results"]}
        retrieved_ids = {m["id"] for m in bundle["retrieved"]["results"]}
        for condition, status in conditions.items():
            result = json.loads((OUT/case_id/condition/"result.json").read_text())
            rows.append({"case_id": case_id, "condition": condition,
                         "target_revenue_forecast_to_actual_error": False,
                         "target_revenue_period_error": False,
                         "target_comparison_coverage": status,
                         "memory_forecast_preserved": True,
                         "stored_items": len(all_ids), "retrieved_items": len(retrieved_ids),
                         "retrieves_entire_store": all_ids == retrieved_ids,
                         "history_tokens": result["history_tokens"], "current_tokens": result["current_tokens"],
                         "evidence_file": str(OUT/case_id/condition/"result.json")})
    summary = {"completed_cases": 3, "completed_answers": len(rows), "repeats_per_condition": 1,
               "claim": "No target revenue forecast-to-fact or target-period error observed in this smoke.",
               "not_claimed": "Not a zero factual-error rate for whole reports, not absence of the phenomenon generally, not causal evidence.",
               "review_method": "Root plus two independent assistant reviewers; no human annotation and no blinded automatic judge.",
               "mean_history_tokens": {c: statistics.mean(r["history_tokens"] for r in rows if r["condition"] == c)
                    for c in ["full_history", "no_memory", "mem0_top10", "mem0_all"]},
               "mean_current_tokens": statistics.mean(r["current_tokens"] for r in rows), "rows": rows,
               "other_flagged_claims": [{"case_id": "nvidia_2022_q2", "condition": "mem0_all",
                    "claim": "$1.22 billion inventory-related charges attributed to the supplied gross-profit reconciliation",
                    "label": "unsupported_by_supplied_evidence_and_false_source_attribution",
                    "not_claimed": "Not necessarily a false real-world number; not established as caused by Memory or forecast-to-fact."}]}
    (OUT/"reviewed_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in {"rows", "other_flagged_claims"}}, indent=2))


if __name__ == "__main__":
    main()
