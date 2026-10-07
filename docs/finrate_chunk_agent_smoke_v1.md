# FIN-RATE naive chunk agent — smoke test

- 질문: 기존 고정 LT 10 + EC 10 중 각 유형의 첫 문항 1개씩. `qa_2_002243`, `qa_3_001181`.
- Corpus: 전체 43개 기업, 15,311개 원래 chunk. 기존 CPU BM25 인덱스 재사용.
- 답변 / judge: `gpt-6-luna` / `gpt-6-luna`, reasoning `medium`.
- 상태: smoke 2문항 실행. 전체 20문항 실행과 구분.

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

- 실행기: [run_finrate_chunk_agent_smoke_v1.py](/home/hob/workspace/asset/replicate/method/run_finrate_chunk_agent_smoke_v1.py)
- 도구: [finrate_chunk_agent_tools_v1.py](/home/hob/workspace/asset/replicate/method/finrate_chunk_agent_tools_v1.py)
- 연결: [finrate_chunk_agent_transport_v1.py](/home/hob/workspace/asset/replicate/method/finrate_chunk_agent_transport_v1.py)
- 결과: [smoke 결과 디렉토리](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent_smoke_v1)

도구 연결은 [공식 OpenAI App Server 문서의 dynamic tool 호출 방식](https://learn.chatgpt.com/docs/app-server#dynamic-tool-calls-experimental)과 로컬 CLI 0.160.1의 생성된 protocol schema를 확인해 구현했다.

## 실행 후 상태

v1은 도구 연결 실패를 보존한 기록이다. 원 응답과 judge를 성능 결과로 사용하지 않는다. 연결 수정 후 같은 두 문항으로 완료한 [v2 설명](/home/hob/workspace/asset/replicate/docs/finrate_chunk_agent_smoke_v2.md)을 참조한다.
