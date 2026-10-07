"""Verify completed paired gold-context runs and compare their original verdicts.

Read-only with respect to both experiments' inputs, responses and assessments.
Writes derived comparison/validation files only in the new run directory.
"""
from collections import Counter
from datetime import datetime
import csv
import io
import json
from pathlib import Path

import run_finrate_gold20_gpt56_luna_judge6_v1 as runner


def load(path):
    return json.loads(path.read_text())


def normalized_argv(argv):
    result = list(argv)
    for flag in ("-C", "-m"):
        result[result.index(flag) + 1] = f"<{flag}>"
    return result


def main():
    new_root, old_root = runner.RUN, runner.PREVIOUS_RUN
    manifest = load(new_root / "manifest.json")
    assert runner.sha(Path(runner.__file__).read_bytes()) == manifest["code_sha256"]
    assert runner.sha((old_root / "manifest.json").read_bytes()) == manifest["previous_manifest_sha256"]
    assert runner.sha((old_root / "report.json").read_bytes()) == manifest["previous_report_sha256"]
    assert runner.sha((runner.COHORT / "inputs.json").read_bytes()) == manifest["cohort_sha256"]
    assert runner.sha((runner.COHORT / "gold_annotations.json").read_bytes()) == manifest["annotations_sha256"]
    previous, current = load(old_root / "report.json"), load(new_root / "report.json")
    assert previous["state"] == current["state"] == "completed"
    old = {r["q_id"]: r for r in previous["rows"]}
    new = {r["q_id"]: r for r in current["rows"]}
    assert set(old) == set(new) == set(manifest["selected_ids"])
    gold = {r["q_id"]: r for r in load(runner.COHORT / "gold_annotations.json")}
    judge = runner.load_judge()
    counts, usage, per_stage, times, paired = Counter(), Counter(), {}, [], []
    flags = []
    for qid in manifest["selected_ids"]:
        a, b = old[qid], new[qid]
        assert a["status"] == b["status"] == "graded"
        assert a["task"] == b["task"] and a["gold_doc_ids"] == b["gold_doc_ids"]
        for stage, model in (("answer", runner.MODEL), ("judge", runner.JUDGE_MODEL)):
            folder = new_root / qid / stage
            response = load(folder / "response.json")
            attempts = sorted(folder.glob("attempt_*"))
            accepted = folder / f"attempt_{response['attempt']:02d}"
            request = load(accepted / "request.json")
            events = [json.loads(line) for line in (accepted / "events.jsonl").read_text().splitlines()]
            valid, messages, reported_usage, _, forbidden = runner.inspect_events(events)
            assert valid and not forbidden and response["returncode"] == 0 and response["error"] is None
            assert response["text"] == messages[0]
            assert request["model"] == response["model"] == model
            assert request["argv"][request["argv"].index("-m") + 1] == model
            assert request["reasoning_effort"] == response["reasoning_effort"] == "medium"
            assert request["prompt_sha256"] == response["prompt_sha256"] == runner.sha((folder / "prompt.txt").read_bytes())
            assert response["usage"] == reported_usage
            old_request = load(old_root / qid / stage / "attempt_01/request.json")
            assert normalized_argv(request["argv"]) == normalized_argv(old_request["argv"])
            if stage == "answer":
                assert (folder / "prompt.txt").read_bytes() == (old_root / qid / stage / "prompt.txt").read_bytes()
                assert response["text"] == b["answer"]
            else:
                ref = gold[qid]
                expected = judge(ref["question"], ref["answer"], b["answer"], ref["key_points"])
                assert (folder / "prompt.txt").read_text() == expected
                assert runner.parse_judge(response["text"], len(ref["key_points"]))["verdict"] == b["verdict"]
            counts[stage] += 1
            for attempt in attempts:
                req, out = load(attempt / "request.json"), load(attempt / "outcome.json")
                counts["attempts"] += 1
                for u in out["usage"]:
                    usage.update(u)
                    per_stage.setdefault(stage, Counter()).update(u)
                times.extend([(datetime.fromisoformat(req["started"]), 1), (datetime.fromisoformat(out["finished"]), -1)])
        paired.append({"q_id": qid, "task": b["task"],
                       "gpt6_verdict": a["verdict"], "gpt56_verdict": b["verdict"],
                       "gpt6_correct": int(a["verdict"] == "CORRECT"),
                       "gpt56_correct": int(b["verdict"] == "CORRECT"),
                       "correct_delta_gpt56_minus_gpt6": int(b["verdict"] == "CORRECT") - int(a["verdict"] == "CORRECT"),
                       "gpt6_key_point_coverage": a["key_point_coverage"],
                       "gpt56_key_point_coverage": b["key_point_coverage"]})
        for model, row in (("gpt-6-luna", a), ("gpt-5.6-luna", b)):
            kp = row["key_point_counts"]
            if row["verdict"] == "CORRECT" and any(kp.get(k, 0) for k in ("partial", "missing", "incorrect")):
                flags.append({"q_id": qid, "answer_model": model, "issue": "CORRECT verdict with incomplete/incorrect key-point labels", "counts": kp})
    assert counts["answer"] == counts["judge"] == 20
    assert dict(usage) == current["provider_usage"]
    active = peak = 0
    for _, delta in sorted(times):
        active += delta
        peak = max(peak, active)
    assert active == 0 and peak <= runner.WORKERS
    tasks = {}
    for task in ("LT-QA", "EC-QA", "ALL"):
        rows = [p for p in paired if task == "ALL" or p["task"] == task]
        n, old_correct, new_correct = len(rows), sum(p["gpt6_correct"] for p in rows), sum(p["gpt56_correct"] for p in rows)
        pairs = Counter((p["gpt6_correct"], p["gpt56_correct"]) for p in rows)
        tasks[task] = {"n": n, "gpt6_correct": old_correct, "gpt56_correct": new_correct,
                       "gpt6_accuracy": old_correct / n, "gpt56_accuracy": new_correct / n,
                       "delta_percentage_points_gpt56_minus_gpt6": 100 * (new_correct - old_correct) / n,
                       "both_correct": pairs[1, 1], "gpt6_only_correct": pairs[1, 0],
                       "gpt56_only_correct": pairs[0, 1], "neither_correct": pairs[0, 0],
                       "gpt6_labels": dict(Counter(p["gpt6_verdict"] for p in rows)),
                       "gpt56_labels": dict(Counter(p["gpt56_verdict"] for p in rows))}
    validation = {"checked": runner.now(), "passed": True, "counts": dict(counts),
                  "peak_concurrent_requests": peak, "provider_usage": dict(usage),
                  "stage_usage": {k: dict(v) for k, v in per_stage.items()},
                  "answer_prompts_byte_identical": 20, "judge_prompt_checks": 20,
                  "old_report_and_manifest_unchanged": True, "unresolved_execution_errors": 0}
    comparison = {"previous_run": str(old_root), "current_run": str(new_root),
                  "judge_model": runner.JUDGE_MODEL, "reasoning_effort": "medium",
                  "metric": "Original judge VERDICT == CORRECT; PARTIAL is not correct",
                  "tasks": tasks, "paired_rows": paired, "automatic_grading_flags": flags,
                  "limitations": ["20 fixed questions, one answer and one judge sample per model; no causal or general model-ranking claim",
                                  "Single GPT-6 Luna judge; prior reference/judging issues retained; no gold or verdict overrides",
                                  "Reasoning setting is matched, actual tokens can differ; CLI/provider output defaults"]}
    runner.write(new_root / "validation.json", validation)
    runner.write(new_root / "comparison.json", comparison)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
    (new_root / "paired_comparison.csv").write_text(out.getvalue())
    lines = ["FIN-RATE gold20: GPT-6 Luna vs GPT-5.6 Luna 답변 비교", "Judge는 모두 GPT-6 Luna/medium. 동일20질문/전체gold/동일answer prompt, worker20.", "정답은 각 최초 judge의 CORRECT만 집계; gold/판정 임의 수정 없음."]
    for task, t in tasks.items():
        lines.append(f"{task}: GPT-6 {t['gpt6_correct']}/{t['n']} ({t['gpt6_accuracy']:.0%}); GPT-5.6 {t['gpt56_correct']}/{t['n']} ({t['gpt56_accuracy']:.0%}); 차이(5.6−6) {t['delta_percentage_points_gpt56_minus_gpt6']:+g}%p")
        lines.append(f"  둘다정답 {t['both_correct']}; 6만정답 {t['gpt6_only_correct']}; 5.6만정답 {t['gpt56_only_correct']}; 둘다비정답 {t['neither_correct']}")
    lines.extend([f"신규 호출 {counts['attempts']}; 미해결 실행오류0; 실제 최대동시요청 {peak}.",
                  "문항별 비교: paired_comparison.csv; 원답변/전체 judge 응답: report.json 및 문항별 폴더.",
                  "20개 표본/단일 judge/1회응답 비교다. 기존 reference·judge 오류를 포함하며 확정된 일반 모델 우열을 뜻하지 않는다."])
    (new_root / "comparison.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"validation": validation, "tasks": tasks, "flags": flags}, ensure_ascii=False))


if __name__ == "__main__":
    main()
