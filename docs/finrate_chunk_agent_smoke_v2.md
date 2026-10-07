# FIN-RATE naive chunk agent — smoke test v2

- 질문: 기존 고정 LT 10 + EC 10 중 각 유형의 첫 문항 1개씩. `qa_2_002243`, `qa_3_001181`.
- Corpus: 전체 43개 기업, 15,311개 원래 chunk. 기존 CPU BM25 인덱스 재사용.
- 답변 / judge: `gpt-6-luna` / `gpt-6-luna`, reasoning `medium`.
- 상태: smoke 2문항 답변·채점·trace 검증 완료. 전체 20문항 평가는 미실행.

## 구조

`질문 → 모델의 도구 선택 → 도구 결과 → 모델의 다음 행동 또는 최종 답변 → 공식 judge`

모델이 검색어·도구·탐색 순서·재검색·종료를 결정한다. 기업별/연도별 처리 순서, 필수 계획·검토 단계, 자동 query 분해·보강, gold 기반 라우팅, bank는 없다. 일반 Codex 작업 지시 대신 짧은 금융 QA 지시를 명시하고, 외부 자료·셸·앱 없이 아래 세 도구만 연결한다.

| 도구 | 동작 |
|---|---|
| `search(query, top_k=10, offset=0)` | 원래 chunk 본문 BM25. ID·제목·점수·검색어 주변 발췌. 모델이 검색어와 top-k를 선택하며 순위 페이지 이동 가능. |
| `grep(pattern, doc_ids, regex, case_sensitive, offset, limit)` | 전체 또는 지정 chunk의 제목·본문에서 문자열/정규식 검색. 일치 위치·주변 본문·페이지 정보 반환. |
| `read(doc_id, start, length)` | 원래 본문의 문자 위치 기반 열람. 응답의 `next_start`로 계속 읽기 가능. |

자료의 날짜·회계기간·기업 관계를 별도 생성하거나 정답에 맞춰 구조화하지 않는다. 제목은 원 배포 제목 그대로다. 검색 발췌도 원문 구간이며 LLM 요약이 아니다.

## 실행 상한

- 모델 응답 최대 20회, 도구 최대 60회, 질문당 최대 15분.
- `read` 한 번의 본문 최대 8,000 추정 tokens. 도구 JSON 전체 최대 10,000 추정 tokens.
- `search`: 기본 top-10, 최대 50; `grep`: 기본 20개 일치, 최대 50. 페이지 이동 가능.
- Context 설정 1,050,000; 실제 provider context 값 별도 기록. 자동 compaction은 정상 결과로 수용하지 않음.
- 상한 소진은 중단/실패로 기록. 별도 마무리 프롬프트나 정답 재생성 없음.
- 로컬 CPU 사용. 디스크 여유 10 GiB 미만이면 중지.

## 보존하는 trace

각 질문의 `agent/`:

- `instructions.txt`, `question.txt`, `tools.json`, `policy.json`, `request_config.json`
- `requests.jsonl`: 보낸 RPC와 정확한 도구 응답
- `events.jsonl`: provider가 노출한 전체 수신 이벤트
- `tools.jsonl`: 도구명·인자·반환 본문·문자 위치·시간·오류·출력 token 추정치
- `usage.json`: provider token usage의 호출별 기록과 누적값
- `response.json`, `response.txt`: 완료 여부·최종 답변·사용량

`judge/`에는 원래 공식 prompt, raw event, 판정 원문을 보존한다. `evidence_audit.json`은 생성 완료 후에만 gold와 대조한 별도 분석이며 agent 입력이 아니다. Gold chunk의 일부 발췌를 봤다는 사실과 필요한 근거 전체를 확보했다는 주장은 구분한다. Provider가 공개하지 않는 내부 추론은 trace로 복원하지 않는다.

## 파일

- 실행기: [run_finrate_chunk_agent_smoke_v2.py](/home/hob/workspace/asset/replicate/method/run_finrate_chunk_agent_smoke_v2.py)
- 도구: [finrate_chunk_agent_tools_v1.py](/home/hob/workspace/asset/replicate/method/finrate_chunk_agent_tools_v1.py)
- 연결: [finrate_chunk_agent_transport_v2.py](/home/hob/workspace/asset/replicate/method/finrate_chunk_agent_transport_v2.py)
- 결과: [smoke 결과 디렉토리](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent_smoke_v2)

도구 연결은 [공식 OpenAI App Server 문서의 dynamic tool 호출 방식](https://learn.chatgpt.com/docs/app-server#dynamic-tool-calls-experimental)과 로컬 CLI 0.160.1의 생성된 protocol schema를 확인해 구현했다.

## Smoke 결과

| 문항 | Search | Grep | Read | 모델 응답 횟수 | Judge |
|---|---|---|---|---|---|
| LT — PENN 이사회 변화 | 3 | 1 | 1 | 6 | INCORRECT |
| EC — Eagle Materials / UNIFI 투자 비교 | 8 | 4 | 1 | 14 | PARTIAL |

- 초기 연결 실패는 `result/finrate_chunk_agent_smoke_v1/`에 별도 보존. 정상 도구 실행 결과는 v2다.
- 두 문항 모두 예산 소진 없이 모델이 스스로 종료. 도구 처리 오류 0건, 외부 도구·compaction 0건.
- 본 실험 전 인터페이스 점검 대상: 모델이 `|`를 사용하면서 `regex=true`를 생략한 grep 4회에서 빈 결과가 반환됐다. 생성 후 같은 요청을 정규식으로 검사하면 각각 20/5/5/11건이 나온다. 이 사후 결과는 모델에 제공하지 않았다.
- LT: 2023년 chunk 앞 8,000자(도입부·안내·목차 위주)만 read하고 2025년은 검색 발췌만 본 상태로 비교 답변을 작성했다.
- EC: UNIFI의 수치는 확보했지만 Eagle의 금액·전략은 확인하지 못했다고 답했다.
- 정답률 0/2는 두 문항 smoke 판정이며 전체 20문항의 성능 추정치로 사용하지 않는다. Judge 원 판정과 해석상의 의문은 trace review에서 구분한다.
- v2 생성 사용량: input 319,934 / output 1,140 tokens. Judge: input 20,292 / output 820 tokens. 연결 실패 및 별도 연결 probe 비용은 각 원 기록에 보존.

[질문별 도구 인자·반환 요약·실제 답변·판정·진단](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent_smoke_v2/trace_review.md) · [원문/trace/token 검증 결과](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent_smoke_v2/validation.json)
