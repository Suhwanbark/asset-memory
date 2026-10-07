"""Write paired answer-model comparisons after completed generation/judging/audit."""
import argparse
from collections import Counter
import csv
import io

import run_finrate_keyword_agent_model50_v1 as variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6-astra", choices=["gpt-6-astra", "gpt-6.1-sol"])
    args = parser.parse_args()
    variant.configure(args.model)
    r = variant.r
    report = r.read(variant.RUN / "report.json")
    validation = r.read(variant.RUN / "validation.json")
    baseline = r.read(variant.PARENT / "report.json")
    assert report["state"].startswith("completed") and validation["passed"]
    old = {(x["method"], x["q_id"]): x for x in baseline["rows"]}
    new = {(x["method"], x["q_id"]): x for x in report["rows"]}
    assert set(new) == set(old) and len(new) == 100
    pairs = []
    for (method, qid), x in new.items():
        y = old[(method, qid)]
        pairs.append({"method": method, "q_id": qid, "task": x["task"],
                      "luna_verdict": y.get("verdict", "EXECUTION_ERROR"),
                      "new_verdict": x.get("verdict", "EXECUTION_ERROR"), "new_state": x["state"],
                      "luna_artifact_dir": y["artifact_dir"], "new_artifact_dir": x["artifact_dir"]})
    paired = {}
    for method in variant.METHODS:
        rows = [p for p in pairs if p["method"] == method]
        paired[method] = {
            "selected": len(rows), "both_correct": sum(p["luna_verdict"] == p["new_verdict"] == "CORRECT" for p in rows),
            "new_only_correct": sum(p["new_verdict"] == "CORRECT" and p["luna_verdict"] != "CORRECT" for p in rows),
            "luna_only_correct": sum(p["luna_verdict"] == "CORRECT" and p["new_verdict"] != "CORRECT" for p in rows),
            "new_labels": dict(Counter(p["new_verdict"] for p in rows)),
        }
    r.common.write(variant.RUN / "model_comparison.json", {"answer_model": args.model, "judge_model": "gpt-6-luna",
                   "paired": paired, "luna_tasks": baseline["tasks"], "new_tasks": report["tasks"]}, frozen=True)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
    writer.writeheader()
    writer.writerows(pairs)
    r.common.freeze_text(variant.RUN / "model_comparison.csv", stream.getvalue())
    display = "GPT-6 Astra" if args.model == "gpt-6-astra" else "GPT-6.1 Sol"
    lines = [f"# FIN-RATE 50문항 — Luna와 {display} 비교", "",
             "- 질문: 동일 LT-QA 25 + EC-QA 25, 총50문항.",
             "- Corpus: 전체43기업, 원래15,311개 chunk.",
             "- 변경: 답변 모델. Judge는 GPT-6 Luna, reasoning effort는 medium 유지.",
             "- 동일 조건: BM25 top-10 전체 본문·순서·prompt; agent search/grep/read·prompt·예산.",
             "- 실행: worker20, 생성 직후 채점, CPU, 10GiB disk guard, 모든 provider-visible trace·usage 보존.", "",
             "| 방법 | 답변 모델 | LT25 | EC25 | 전체50 |", "|---|---|---|---|---|"]
    for method, label in (("bm25", "BM25 top-10"), ("naive_agent", "Naive agent")):
        for name, source in (("GPT-6 Luna", baseline), (display, report)):
            stats = source["tasks"][method]
            cells = [f"{stats[t]['correct']}/{stats[t]['selected']} ({stats[t]['accuracy']:.0%})" for t in ("LT-QA", "EC-QA", "ALL")]
            lines.append(f"| {label} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "## 판정 및 문항별 비교", "", "정답률은 CORRECT만 정답으로 집계한다. PARTIAL·INCORRECT·FAILURE·실행 오류를 분모에서 제외하지 않는다.", ""]
    for method in variant.METHODS:
        p = paired[method]
        lines.append(f"- {method}: 새 모델만 정답{p['new_only_correct']}, Luna만 정답{p['luna_only_correct']}, 둘 다 정답{p['both_correct']}. 새 판정: {p['new_labels']}.")
    lines += ["", "## 기존 Gold-context 결과 — 별도20문항", "",
              "| 답변 모델 | LT10 | EC10 | 전체20 |", "|---|---|---|---|",
              "| GPT-6 Luna | 2/10 (20%) | 6/10 (60%) | 8/20 (40%) |",
              "| GPT-5.6 Luna | 3/10 (30%) | 6/10 (60%) | 9/20 (45%) |", "",
              "Judge는 둘 다 GPT-6 Luna. Gold chunk 본문만 제공하며 reference answer는 답변 모델에 주지 않았다. Gold50 추가 실행은 앞선 범위 정정 때 중지됐으며, 이번 모델 비교에서 gold 조건을 새로 실행하지 않았다.", "",
              "## 해석과 검증", "",
              "같은 질문·입력·judge 조건의 모델 비교이며, 방법 간 실제 토큰 소비가 동일하다는 뜻은 아니다. 단일 Luna judge를 사용하므로 judge의 한계는 남는다. 성적을 보고 질문을 교체하거나 정상 답변을 재생성하지 않았다.", "",
              f"- 상태: {report['state']}; 채점{report['graded']}/100.",
              f"- 검증: {validation['graded_pairs_verified']}건의 모델·prompt·공식judge·판정 및 원문span{validation['source_spans_verified']}개.",
              f"- 실행 오류: {len(validation['execution_errors'])}건. 원인과 trace는 validation/report에 보존.", "",
              f"[결과]({variant.RUN}/report.json) · [모델별 문항 비교]({variant.RUN}/model_comparison.csv) · [검증 및 사용량]({variant.RUN}/validation.json)", ""]
    path = r.ROOT / f"docs/finrate_keyword_agent50_{args.model.replace('-', '_')}_v1.md"
    r.common.freeze_text(path, "\n".join(lines))
    print(r.json.dumps({"report": str(path), "paired": paired, "scores": report["tasks"]}))


if __name__ == "__main__":
    main()
