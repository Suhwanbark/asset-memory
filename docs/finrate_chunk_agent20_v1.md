# FIN-RATE naive chunk agent — 고정 20문항 결과

- 답변 / judge: `gpt-6-luna` / `gpt-6-luna`, reasoning `medium`.
- 문항: 기존 LT 10 + EC 10. Smoke 2문항 원본 재사용, 나머지 18문항 신규 실행.
- Worker: 20. 생성 완료 직후 공식 judge로 채점.
- Corpus: 전체 43개 기업 / 15,311개 원래 chunk.
- 도구·prompt·예산: 검증된 smoke v2와 동일.
- 완료: 20/20 답변 및 채점. 중단·미채점 0, 재시도 0.

| 유형 | CORRECT | PARTIAL | INCORRECT | 정답률 |
|---|---:|---:|---:|---:|
| LT-QA | 3 | 2 | 5 | 30% |
| EC-QA | 2 | 5 | 3 | 20% |
| ALL | 5 | 7 | 8 | 25% |

## 같은 20문항 비교

| 방법 | LT-QA | EC-QA | 전체 |
|---|---:|---:|---:|
| BM25 top-10 | 20% | 0% | 10% |
| Naive chunk agent | 30% | 20% | 25% |
| Gold-context / GPT-6 Luna | 20% | 60% | 40% |

BM25 대비 agent만 정답인 문항은 5개, BM25만 정답인 문항은 2개다. 전체 점수 상승이 모든 문항의 개선을 뜻하지 않는다. 입력량·반복 호출 예산이 같은 비교도 아니다.

## Trace 점검

- 모델 응답 164회, 도구 호출 156회: search 90 / grep 31 / read 35.
- 모델 응답 최대 16회로 모든 문항이 20회 상한 전에 스스로 종료.
- 정규식 옵션 없이 `|`를 사용한 grep: 7문항에서 12회.
- 도구 오류 1건: `qa_3_000381`에서 존재하지 않는 `doc_003`을 지정. 오류를 반환한 뒤 agent가 계속 진행해 최종 답변과 채점을 완료.
- 원문 구간 1,435개와 도구 반환·문자 위치 일치 확인.
- Smoke 2문항 파일 해시·고정 질문·코드·모델/effort·공식 judge prompt·token 합계 검증 통과.

## Provider token 사용량

| 구분 | Input | Output |
|---|---:|---:|
| Agent | 2,414,722 | 11,525 |
| Judge | 204,715 | 8,333 |

Smoke 2문항의 기존 사용량을 포함한다. 초기 연결 실패와 별도 synthetic probe 비용은 각 원 실행 기록에 별도 보존한다.

## 파일

- [원 결과](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent20_gpt6_luna_v1/report.json)
- [문항별 비교](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent20_gpt6_luna_v1/paired_comparison.csv)
- [Trace 진단](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent20_gpt6_luna_v1/trace_diagnostics.json)
- [검증](/home/hob/workspace/asset/replicate/result/finrate_chunk_agent20_gpt6_luna_v1/validation.json)

각 `q_id/agent/`에 요청·이벤트·도구 인자/반환·모델 응답·usage, `q_id/judge/`에 공식 채점 prompt와 응답을 보존한다.
