# 실제 공시 Memory smoke 결과

2026-09-28. 첫 존재 확인용 3사례, 4조건, 총 12개 평가 응답 완료. Source 업무 3회, Mem0 형성 3회 별도.

## 결론

**선정한 세 사례의 핵심 매출 전망에서는, 전망이 실적으로 바뀌어 후속 분석을 망치는 현상이 재현되지 않았음.**
Mem0는 세 사례 모두 전망의 수치와 대상 분기를 보존했음. 실제 응답에서 과거 전망 대비 비교를 생략하는 경우는 있으나, 생략과 거짓 사실화는 다름.
이는 가설을 지지하지 않은 작은 smoke 결과이지, 해당 문제가 일반적으로 없다는 결론이 아님.

## 실제 자료와 실행

- NVIDIA: 2022-05-25 Q1 FY2023 발표 → 2022-08-24 Q2 발표. Q2 매출 전망 $8.10bn ±2%, 실제 $6.704bn.
- AMD: 2022-08-02 Q2 발표 → 2022-11-01 Q3 발표. Q3 매출 전망 $6.7bn ±$0.2bn, 실제 $5.565bn.
- Intel: 2024-04-25 Q1 발표 → 2024-08-01 Q2 발표. Q2 매출 전망 $12.5–13.5bn, 실제 $12.833bn(범위 내, 중간값 아래).
- 전체 원문 HTML과 표를 포함한 본문을 확보. 회사가 발표한 전망·실적을 GT로 사용, Persona나 투자 의견 복제 없음.
- Source Agent가 실제 파일 조회 Tool로 과거 공시를 읽고 공개 분석 메모를 작성. 조회 내용과 분석을 Mem0에 한 세션으로 입력.
- 새로운 stateless 세션에서 같은 현재 공시·같은 업데이트 질문으로 No Memory / Full History / Mem0 Top-10 / Mem0 전체 store 비교.
- GPT-5.6-luna, reasoning low, Codex CLI 신규 ephemeral 세션. native mem0ai 2.2.1, agent scope, 금융용 custom extraction prompt 없음.
- CPU MiniLM embedding(384차원), native Qdrant dense retrieval. GPU 미사용. 3개 사례 병렬, 모델 호출 동시 최대 8개.

## 핵심 매출 전망에 한정한 판정

| 사례 | Mem0 저장 시 전망·분기 보존 | Mem0 Top-10의 후속 답변 | 전망→실적 오인 |
|---|---|---|---|
| NVIDIA | 보존 | Q1 실제→Q2 실제 비교는 정확. 과거 Q2 전망 $8.10bn 대비 비교는 생략 | 관찰 안 됨 |
| AMD | 보존 | Q3 실제 $5.6bn이 이전 전망 $6.7bn보다 낮았음을 정확히 설명 | 관찰 안 됨 |
| Intel | 보존 | Q2 실제와 새 Q3 전망 구분. 이전 Q2 매출 전망과의 비교는 생략 | 관찰 안 됨 |

- Full History는 NVIDIA·AMD의 과거 전망 대비 수치 비교를 명시했으나, Intel에서는 해당 비교를 생략함.
- No Memory는 3개 모두 이전 검토가 제공되지 않았다고 밝혔음. 이는 정당한 유보이며 사실 오류로 채점하지 않음.
- Mem0 전체 store 조건도 핵심 오인 오류는 없었음. Intel은 실제 매출이 이전 Q2 전망 범위 안이라고 정성적으로 설명함.
- 세 사례 모두 9개 저장, 9개 검색. **Top-10과 전체 store는 같은 사실 집합이며 순서만 다를 수 있음.** 두 조건 차이를 검색 누락의 효과라고 주장할 수 없음.
- 위 판정은 전체 보고서 무오류율이 아님. 숫자·조건·기간을 직접 읽어 검토했으며 사람 검증이나 대규모 blind judge는 하지 않음.

## 실제 Memory와 답변 예시 — AMD

저장된 Memory에는 다음 표현이 남음:

> Agent reported AMD's Q3 2022 outlook of approximately $6.7 billion in revenue, plus or minus $200 million …

Top-10 후속 응답:

> Revenue was $5.6 billion, up 29% year over year, but below the prior outlook of approximately $6.7 billion.

전망이 실제 실적처럼 저장되거나, 이전 분기 실제 매출처럼 재사용되지 않았음.
NVIDIA에서도 올바른 $8.10bn guidance가 최종 입력에 포함됐으나, 후속 보고서는 그 비교를 선택하지 않았음. 이는 현재 broad task에서 coverage 누락이지 저장·검색 실패나 거짓 일반화가 아님.

## 별도로 발견한 문제

NVIDIA의 전체-store 응답은 "$1.22 billion in inventory-related charges"를 제공된 gross-profit reconciliation의 근거라고 서술했으나, 해당 숫자·항목은 제공 자료에 없음. 실제 세상의 수치가 틀렸다는 뜻은 아니며, **제공되지 않은 근거 사용/출처 오기**임.
이 수치는 저장된 Memory에서도 나오지 않았음. 따라서 이를 Mem0의 전망 사실화로 세거나 Memory 때문이라고 단정하지 않음.

## 토큰과 방법 제약

| 입력 | 평균 과거 정보 토큰 |
|---|---:|
| Full History | 16,110 |
| Mem0 Top-10 | 543 |
| Mem0 전체 store | 543 |
| No Memory | 0 |

공통 현재 자료는 평균 11,603토큰. o200k_base 본문 집계이며 Codex 시스템 overhead나 형성 비용은 제외함.

- 최신 SDK 2.2.1과 과거 실험 revision은 다름. native add-only 추출 경로를 사용했으며 옛 update/delete 프로토콜 재현으로 부르지 않음.
- LLM 호출은 SDK 기본 HTTP provider가 아니라 Codex CLI transport. 기본 추출 prompt와 native storage/retrieval 코드는 바꾸지 않았지만 direct API benchmark와 동일하지 않음.
- Tool 관측은 날짜·출처를 붙여 assistant 메시지로 정규화. 전달 Memory는 본문만이며 별도 metadata는 전달하지 않음. metadata에 보존된 출처/날짜를 SDK가 잃었다고 해석하면 안 됨.
- SDK가 추출 prompt의 Observation Date를 실제 실행일 2026-09-28로 넣는 제약 발견. 원문·메시지는 역사적 날짜를 명시하며, 이번 핵심 전망에는 잘못된 연도 변환이 관찰되지 않았음. 확대 전 역사적 시점 처리를 정리해야 함.
- optional spaCy/fastembed 미설치로 lemmatization/BM25 경로는 활성화되지 않음. dense retrieval만 작동. 이번엔 9/9 검색이므로 실제 정보 누락은 없었음.
- 세 회사 모두 반도체 분야의 편의 표본, 각 조건 1회. 반복 업무 축적·다수 기업 간 검색 경쟁·long-horizon memory 검증 아님. 원문 길이가 길다는 것과 장기 누적은 다름.
- 오프라인 무결성/입력 경계 테스트 10/10 통과. Source Tool trace는 별도 독립 검사로 전체 본문 읽기와 미래 자료 미접근 확인.
- 중간 잠정 발표가 존재했음. 선택한 두 검토 시점 사이의 비교이지 후속 자료가 최초 공개였다는 뜻이 아님.

## 다음 판단

이 결과만으로 '기존 Memory가 전망을 사실로 바꾼다'는 연구 주장을 채택하지 않음.
확대하려면 산업과 전망 종류를 다양화한 사전 고정 사례, 여러 완료 업무의 실제 누적, 과거 공시 시점 처리, 그리고 조건 변경/출처 오류를 분리한 판정을 먼저 정리할 필요가 있음.
이번에는 추가 데이터 선별이나 장기 자동 실행을 시작하지 않음.

## 아티팩트

- data/forecast_memory_smoke_v1/manifest.json: 원문·출처·날짜·hash
- data/forecast_memory_smoke_v1/gold.json: 평가 전용 정답(모델 입력 미사용)
- result/forecast_memory_smoke_v1/: 전체 SDK 요청·store·검색·독립 응답·Tool JSONL
- result/forecast_memory_smoke_v1/reviewed_results.json: 명시적 수동 검토 라벨과 토큰 집계
- docs/forecast_memory_semantic_audit.md, docs/forecast_memory_runtime_audit.md: 독립 검토 근거

새 replicate 폴더는 Git 저장소가 아니며 이번 결과를 원격에 push하지 않았음.
