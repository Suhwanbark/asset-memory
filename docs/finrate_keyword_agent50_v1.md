# FIN-RATE Keyword·naive agent — 50문항

- 문항: 고정 무작위 표본 순서의 LT-QA 25 + EC-QA 25. 기존 20문항 포함.
- Corpus: 43개 기업, 15,311개 원래 chunk 전체.
- 답변 모델 / Judge: GPT-6 Luna / GPT-6 Luna, medium.
- 평가: FIN-RATE 공식 judge prompt. 정답률 = CORRECT / 50.
- 실행: worker 20에서 생성 직후 채점. CPU, 여유 공간 10GiB 미만 중지.
- 범위: Keyword BM25와 naive chunk agent. Gold-context 추가 실행은 사용자 정정 후 중지·보존.

| 방법 | LT-QA | EC-QA | 전체 | CORRECT | PARTIAL | INCORRECT | FAILURE |
|---|---|---|---|---|---|---|---|
| BM25 top-10 | 5/25 (20%) | 0/25 (0%) | 5/50 (10%) | 5 | 25 | 19 | 1 |
| Naive agent | 3/25 (12%) | 2/25 (8%) | 5/50 (10%) | 5 | 25 | 20 | 0 |

FAILURE 1건은 근거 부족으로 비교를 수행하지 않은 답변에 대한 공식 judge 판정이다. 호출 실패가 아니며 50문항 분모에 포함했다. 미해결 실행 오류는 0건이다.

## 설정

- BM25: 전체 원래 chunk 본문 검색, top-10 본문 전체 제공. 재분할·요약·회사/연도 필터·rerank 없음.
- Naive agent: 같은 corpus에서 `search` / `grep` / `read`. 검색어·도구·읽을 구간·종료는 agent가 결정한다. 기존20과 같은 도구·prompt·예산, `grep(regex=False)` 기본값 유지.
- Agent 한도: 모델 응답20, 도구60, 900초. `read` 한 번에 본문 최대8,000 token, 후속 offset으로 계속 읽기 가능.
- 모든 조건에 같은 질문·모델·rubric을 적용했다. 두 방식의 총 입력 토큰은 같지 않다.

## 재사용 및 검증

- 3조건 확대 실행 중 사용자 범위 정정으로 중지. 기존 결과·중단된 trace·gold 부분 결과를 원본 경로에 보존했다.
- 범위 정정 시점에 완료된 62개 답변·채점 재사용. 연결 재시도/HTTPS fallback 뒤 정상 완료된 응답·채점 16개를 원래 trace에서 복구했다. 나머지 누락 단계를 실행했다.
- 복구는 모델·prompt·완료 turn·usage·금지 도구 없음 검증 후 수행했다. 완료 답변의 내용이나 점수로 재시도·선택하지 않았다. 기존 오류 기록과 모든 시도는 보존했다.
- 최종100건의 질문·전체 본문·모델 설정·공식 judge·성능 집계 검증 통과.
- Agent 도구390회: search219 / grep68 / read103. 반환된 원문 구간3,865개를 원문 offset과 대조했다.
- 정답 문항은 서로 겹치지 않는다: BM25만 정답5개, agent만 정답5개. 이 표본에서는 전체 정답률 우위를 보이지 않았다.

## 토큰 사용량

선택된 최종 응답·채점의 provider 보고 사용량이며, 기존20문항 재사용분을 포함한다. Cached input은 input의 일부다.

| 방법 | 답변 input | 답변 output | Judge input | Judge output |
|---|---:|---:|---:|---:|
| BM25 | 13,672,241 | 10,677 | 512,260 | 20,742 |
| Naive agent | 6,357,200 | 29,895 | 515,249 | 21,811 |

중단된 시도까지 합산하면 agent 답변 input 7,114,062 / output 32,271이다. 모든 시도의 provider 보고 사용량과 중복 제거된 trace 경로는 `usage_all_attempts.json`에 보존했다.

## 기록

- [문항 목록](/home/hob/workspace/asset/replicate/data/finrate_baseline50_v1/inputs.json)
- [결과 및 문항별 artifact 경로](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt6_luna_v1/report.json)
- [문항별 비교](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt6_luna_v1/paired_comparison.csv)
- [검증](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt6_luna_v1/validation.json)
- [전체 시도 사용량](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt6_luna_v1/usage_all_attempts.json)
- [Agent 탐색 집계](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt6_luna_v1/trace_diagnostics.json)

각 문항의 `agent/`에 requests·events·tools·usage·최종 답변, `answer/`에 BM25 입력·응답, `judge/`에 judge prompt·응답이 있다. 재사용분은 원본을 가리키는 symlink로 보존한다. Trace 범위는 provider가 노출한 이벤트이며 숨겨진 내부 추론을 포함한다는 뜻은 아니다.
