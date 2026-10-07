"""Audit fixed inputs, original-source tool outputs, usage and judge continuity."""
from collections import Counter
import json
from pathlib import Path

import finrate_bm25_top10_v1 as bm25
from finrate_chunk_agent_tools_v1 import ChunkTools
import run_finrate_gold20_gpt56_luna_judge6_v1 as common

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "result/finrate_chunk_agent_smoke_v2"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    questions = json.loads((ROOT / "data/finrate_chunk_agent_v1/inputs.json").read_text())
    original = json.loads((ROOT / "data/finrate_gold20_gpt6_luna_v1/inputs.json").read_text())
    assert questions == [{k: q[k] for k in ("q_id", "task", "question")} for q in original]
    assert Counter(q["task"] for q in questions) == {"LT-QA": 10, "EC-QA": 10}
    assert manifest["smoke_ids"] == [next(q["q_id"] for q in questions if q["task"] == task) for task in ("LT-QA", "EC-QA")]
    for name, digest in manifest["source_sha256"].items():
        assert bm25.sha_file(RUN / "source_snapshot" / name) == digest
        assert bm25.sha_file(ROOT / "method" / name) == digest
    assert bm25.sha_file(common.REFERENCE / "evaluation/qa_llm_judge.py") == manifest["judge_source_sha256"]
    docs, corpus_sha = bm25.load_corpus()
    assert corpus_sha == manifest["corpus_zip_sha256"]
    source = {d["_id"]: d for d in docs}
    env = ChunkTools(docs, None)  # Only grep is needed for offline diagnostics.
    gold = {q["q_id"]: q for q in json.loads((ROOT / "data/finrate_gold20_gpt6_luna_v1/gold_annotations.json").read_text())}
    summary = []
    totals = {"agent_input_tokens": 0, "agent_output_tokens": 0, "judge_input_tokens": 0, "judge_output_tokens": 0}
    outline = ["# Naive chunk agent smoke trace", "", "고정20 중 첫 LT/EC 각1개. 아래 진단은 답변 생성 후 수행했으며 모델에 돌려주지 않았다.", ""]
    for qid in manifest["smoke_ids"]:
        folder = RUN / qid
        question = next(q for q in questions if q["q_id"] == qid)
        response = json.loads((folder / "agent/response.json").read_text())
        assert response["state"] == "completed" and not response["error"] and not response["forbidden_items"]
        assert response["tool_calls"] > 0
        assert response["model"] == "gpt-6-luna" and response["reasoning_effort"] == "medium"
        requests = jsonl(folder / "agent/requests.jsonl")
        events = jsonl(folder / "agent/events.jsonl")
        calls = jsonl(folder / "agent/tools.jsonl")
        actual_thread = next(x["result"] for x in events if x.get("id") == 2 and "result" in x)
        assert actual_thread["model"] == "gpt-6-luna" and actual_thread["reasoningEffort"] == "medium"
        start = next(x for x in requests if x.get("method") == "turn/start")
        assert start["params"]["input"] == [{"type": "text", "text": question["question"]}]
        thread = next(x for x in requests if x.get("method") == "thread/start")
        assert thread["params"]["baseInstructions"] == manifest["instructions"]
        assert thread["params"]["dynamicTools"] == manifest["tools"]
        assert thread["params"]["config"]["features.code_mode_host"] is True
        server_calls = [e for e in events if e.get("method") == "item/tool/call"]
        completed = [e["params"]["item"] for e in events if e.get("method") == "item/completed"
                     and e["params"]["item"]["type"] == "dynamicToolCall"]
        assert len(calls) == len(server_calls) == len(completed) == response["tool_calls"]
        diagnostics, span_count = [], 0
        outline += [f"## {question['task']} — {qid}", "", question["question"], "",
                    "| 순서 | 도구 | 인자 | 반환 |", "|---|---|---|---|"]
        for call in calls:
            observed = next(e["params"] for e in server_calls if e["params"]["callId"] == call["call_id"])
            delivered = next(e for e in completed if e["id"] == call["call_id"])
            assert observed["arguments"] == call["arguments"] and observed["tool"] == call["tool"]
            assert json.loads(delivered["contentItems"][0]["text"]) == call["result"]
            result = call["result"]
            if call["tool"] == "search":
                spans = [s for row in result["results"] for s in row["excerpts"]]
                returned = f"{len(result['results'])} chunks"
            elif call["tool"] == "grep":
                spans = result["matches"]
                returned = f"{len(spans)} matches"
                if "|" in call["arguments"]["pattern"] and not call["arguments"].get("regex", False):
                    corrected = env.grep(**{**call["arguments"], "regex": True})
                    diagnostics.append({"call_index": call["index"], "type": "regex_flag_omitted",
                                        "original_matches": len(spans), "offline_regex_matches": len(corrected["matches"]),
                                        "not_shown_to_model": True})
            else:
                spans = [result]
                returned = f"{result['doc_id']} chars {result['start']}–{result['end']}"
            for span in spans:
                assert span["text"] == source[span["doc_id"]][span["field"]][span["start"]:span["end"]]
                span_count += 1
            args = json.dumps(call["arguments"], ensure_ascii=False).replace("|", "\\|")
            outline.append(f"| {call['index']} | {call['tool']} | `{args}` | {returned} |")
        usage = json.loads((folder / "agent/usage.json").read_text())
        for key in ("inputTokens", "outputTokens", "cachedInputTokens", "reasoningOutputTokens"):
            assert sum(u["last"][key] for u in usage) == response["provider_total"][key]
        assessed = json.loads((folder / "assessment.json").read_text())
        ref = gold[qid]
        points = ref["key_points"] if isinstance(ref["key_points"], list) else [ref["key_points"]]
        assert (folder / "judge/prompt.txt").read_text() == common.load_judge()(question["question"], ref["answer"], response["answer"], points)
        judge = json.loads((folder / "judge/response.json").read_text())
        assert judge["model"] == "gpt-6-luna" and judge["reasoning_effort"] == "medium"
        totals["agent_input_tokens"] += response["provider_total"]["inputTokens"]
        totals["agent_output_tokens"] += response["provider_total"]["outputTokens"]
        totals["judge_input_tokens"] += sum(x["input_tokens"] for x in assessed["judge_usage"])
        totals["judge_output_tokens"] += sum(x["output_tokens"] for x in assessed["judge_usage"])
        summary.append({"q_id": qid, "verdict": assessed["verdict"], "tools": dict(Counter(c["tool"] for c in calls)),
                        "source_spans_verified": span_count, "diagnostics": diagnostics,
                        "model_responses": response["model_responses"], "tool_calls": response["tool_calls"],
                        "provider_total": response["provider_total"]})
        outline += ["", "### 실제 최종 답변", "", response["answer"], "", f"**Judge: {assessed['verdict']}**", "",
                    assessed.get("judge_analysis", ""), "", "### 도구 사용 진단", ""]
        for diagnostic in diagnostics:
            outline.append(f"- 호출 {diagnostic['call_index']}: `regex` 옵션 없이 `|` 사용 → 원래 {diagnostic['original_matches']}건. 답변 후 동일 요청을 정규식으로 조회하면 {diagnostic['offline_regex_matches']}건 반환.")
        if qid == "qa_2_002243":
            outline += ["- `read`는 2023년 chunk의 첫 8,000자 한 번. 이 구간은 도입부·주주총회 안내·목차가 대부분이며, 2025년 chunk는 검색 발췌만 반환됐다.",
                        "- Judge가 ‘majority-independent’와 ‘CEO를 제외하고 독립’ 표현을 충돌로 취급한 부분은 논리적 모순이라고 단정할 수 없다. 원 판정은 유지하며, 구체적 인물·경력 변화 누락과 구분한다."]
        else:
            outline.append("- UNIFI의 2021년 지출은 답했지만 Eagle의 금액·전략은 확인하지 못했다고 답했다. 모델은 14회 응답 후 스스로 종료했으며 20회 상한에 도달하지 않았다.")
        outline.append("")
    validation = {"passed": True, "updated": common.now(), "scope": "smoke2 only", "rows": summary,
                  "totals": totals, "checks": ["same frozen20 inputs", "first question per task", "source snapshots unchanged",
                  "all returned source spans exact", "tool RPC and completed results match", "actual model/effort",
                  "provider usage reconciles", "official judge prompt unchanged", "no compaction or unexpected tools"]}
    common.write(RUN / "validation.json", validation, frozen=True)
    common.freeze_text(RUN / "trace_review.md", "\n".join(outline) + "\n")
    print(json.dumps({"passed": True, "totals": totals, "rows": [{k:v for k,v in row.items() if k != 'provider_total'} for row in summary]},ensure_ascii=False))


if __name__ == "__main__":
    main()
