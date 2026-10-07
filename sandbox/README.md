# sandbox — 격리된 Docker에서 web search agent로 FIN-RATE 질문 답변

질문 1개마다 Docker 컨테이너 1개를 띄워 Codex CLI(`codex exec --search`)가 답변을 생성한다. 컨테이너에는 질문 원문만 들어가며, `q_id`·`task`·gold 필드는 host에만 남는다. 전체 trace를 보존하고, 실행 후 FIN-RATE contamination을 검사한다.

finautorubric(`eval_research/finautorubric`)의 sandbox·session·contamination 코드를 이식했고, 루브릭 생성·리뷰·채점 파이프라인은 포함하지 않는다. 채점은 기존 FIN-RATE judge로 `prediction.json`을 읽어 별도로 한다.

## 요구사항

- Docker Engine
- Python 3.11 이상 (host 코드는 표준 라이브러리만 사용)
- Codex 인증: `~/.codex/config.toml`에 `cli_auth_credentials_store = "file"`을 두고 `codex login` → `~/.codex/auth.json`. API key를 쓸 때는 `--api-key-env OPENAI_API_KEY`.

## 사용법 (repo root에서)

```bash
python -m sandbox build-image --image asset-memory-sandbox:v1
python -m sandbox preflight --image asset-memory-sandbox:v1
python -m sandbox run --inputs data/finrate_baseline50_v1/inputs.json \
  --output-dir result/sandbox_web_gpt6_luna_v1 --model gpt-6-luna --limit 1
python -m sandbox run --inputs data/finrate_baseline50_v1/inputs.json \
  --output-dir result/sandbox_web_gpt6_luna_v1 --model gpt-6-luna
python -m sandbox scan --run-dir result/sandbox_web_gpt6_luna_v1
```

- 기본값: `--concurrency 20`, `--timeout 900`(질문당 wall seconds), `--effort medium`, 컨테이너당 `--cpus 1 --memory-mb 2048`.
- `run`은 시작 전에 preflight를 다시 돌리고, 통과한 image ID로만 컨테이너를 만든다.
- 같은 `--output-dir`로 다시 실행하면 `result.json`이 있는 질문은 건너뛴다. 중단된 질문의 trace는 `_aborted/`로 옮긴 뒤 다시 실행한다. 모델·effort·timeout·image·prompt·차단 목록이 바뀌었으면 실행을 거부한다.
- `scan`은 marker를 바꾼 뒤 기존 run의 contamination만 다시 판정한다.

## 격리

| 항목 | 설정 |
|---|---|
| 파일시스템 | 루트 읽기 전용, `/work`·`/home/agent`·`/tmp`는 익명 볼륨, host bind mount 없음 |
| 권한 | UID 1000, `--cap-drop ALL`, `no-new-privileges`, CPU·메모리·PID·파일 크기 제한 |
| 네트워크 | bridge (모델 API와 웹 검색용). HF·GitHub·arXiv host는 `/etc/hosts`에서 `0.0.0.0` |
| Codex | `--ignore-user-config`, MCP 없음, apps·plugins·browser·memories·multi-agent 등 비활성 |
| 인증 | `auth.json`은 `/home/agent/.codex`에만 주입, trace·archive·manifest에 포함되지 않음 |

## Contamination

실행 후 host에서 `turn_*.jsonl`, `work.tar`, `sessions.tar`를 검사한다(`contamination.py`).

- `blocked`: agent가 만든 문자열(검색어, 셸 명령, `answer.json`의 `sources`)이나 agent가 연 페이지의 URL·제목에 FIN-RATE marker(`fin-rate`, `finrate`, `gglabyale`, `2602.07294`)나 데이터셋 URL이 있음 → `score_eligible=false`.
- `suspect`: 검색 결과 목록에만 노출, 차단 host 요청, 답변·출력 속 marker 노출, 질문 원문 10단어 이상을 그대로 검색 등 → 채점에 포함하되 표시.
- `clean`: 신호 없음.

Codex 0.159.2의 `--json` trace에서 web_search 항목은 `action: search`이면 검색(결과 목록은 `results`), 그 외(`other`)이면 페이지 열람이며 연 페이지가 `results[].url`·`title`에 남는다. `result.json`의 `activity.opened_urls`도 여기서 뽑는다.

한계:

- Codex의 web search는 서버 측에서 실행되므로 `/etc/hosts` 차단이 적용되지 않는다. 이 경로는 실행 후 trace 검사로만 통제한다. Mac smoke에서 데이터셋을 찾으라고 지시한 canary는 셸 접속은 막혔지만 web search로 HF `qa/LT-QA.json` 페이지를 열었고, `page_opened`로 `blocked` 판정됐다.
- 검색 결과 snippet에 정답이 노출되는 경우는 `suspect`로만 남는다.
- 시간 제한만 있고, 기존 naive agent처럼 모델 응답·도구 호출 횟수를 제한하지는 않는다.

## 산출물

```
<output-dir>/
├── manifest.json        # settings(모델, image ID, prompt SHA-256, 차단 목록 등)와 provenance(git, Codex 버전, 입력 SHA-256)
├── preflight.json, answer_prompt.md
├── predictions.jsonl    # q_id, task, state, answer, contamination_status, score_eligible
├── summary.json         # 상태·contamination·task별 집계, 토큰 합계
└── <q_id>/
    ├── input.json, prompt_01.txt
    ├── turn_01.jsonl    # Codex 이벤트 전체: 검색어, 명령과 출력, 메시지, 토큰 사용량
    ├── turn_01.stderr, session.json
    ├── work.tar         # /work 전체 (host에서 압축을 풀지 않음)
    ├── sessions.tar     # Codex 세션 기록 (연 페이지 URL 포함)
    ├── contamination.json
    ├── result.json      # status, answer, sources, usage, activity(opened_urls 포함), elapsed
    └── prediction.json  # 기존 judge runner가 읽는 형식 (state, answer)
```

`status`는 `completed`(정상 종료와 답변), `incomplete`(시간 초과·오류 뒤 남은 답변), `failed`(답변 없음) 중 하나다.
