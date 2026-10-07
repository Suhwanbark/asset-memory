# 실제 금융 사건의 시간별 근거 유효성 POC — 원문 QC 후 29개

2026-09-28. **현재 결론: 이 단순한 설정에서는 ‘낡은 근거를 현재에도 그대로 적용하는’ 핵심 현상이 재현되지 않음.** 요약 후 일부 근거가 빠지는 양상은 있지만, 정보의 현재 효용을 일반적으로 판단하지 못한다는 주장과는 구분해야 함. 최초 30개 실행 후 원문 QC로 1개를 제외하고, 1개의 본문 추출을 수정해 재평가한 **29개 사건·111문항을 최종 집계**함.

## 1. 무엇을 실제로 실행했나?

- 최초 사건 30개: 2023–2025년 정기 FOMC 결정 24회 전수 + 기업 사건 6개. 원문 QC 후 FOMC 24개 + 기업 5개를 유지함.
- 최종 기업 사건: JetBlue–Spirit, Adobe–Figma, Amazon–iRobot, Microsoft–Activision, Kroger–Albertsons. VanEck 미국 현물 Bitcoin ETF 출시 사례는 역사적 원문 파일의 시점 검증 문제로 제외함.
- 근거: 공식 성명·회의록·회사 발표 원문. 가상 대화·조작한 수치·noise 문장은 생성하지 않음.
- FOMC는 직전 3회 성명 + 해당 회의 성명 + 해당 회의록 전체를 제공. 회의일이 아니라 **회의록 공개일 이후**를 cutoff로 설정.
- 최종 111개 판정 문항: 과거 전제가 지금도 적용되는지, 과거의 사실 자체는 여전히 참인지, 지속적인 목표가 남아 있는지, 미래 결정이 실제로 확정됐는지 구분. **111개 독립 사건이 아님.** 최초 실행은 114문항이며 삭제하지 않고 보존함.
- 모델: GPT-5.6 Luna, reasoning low. 최대 8개 사례 병렬, GPU 미사용.
- 조건: 원문 전체 / Naive retrieval / 일반적인 간단 요약. 검색은 220-word 비중첩 청크, word TF-IDF cosine Top-10. Dense RAG, 공식 Mem0/A-MEM 실험이 아님.
- 정답은 평가 시점까지의 공식 기록으로 고정. 최종 시장 결과·가격 변화·Source Agent의 의견으로 정하지 않음.

## 2. 예측시장과의 연결 범위

Polymarket 직접 API가 HTTP451을 반환하여 우회하지 않음. 대신 ForecastBench가 공개 배포한 과거 질문/해결 데이터의 고정 commit을 검토해 금융 관련 후보 126개를 확보함. 시장 ID가 서로 다르다고 독립 사건으로 세지 않음.

그중 **6개 FOMC 시점에서 당시 존재했던 다음 회의 예측 질문 9개**와 연결 가능함. 최종 29개 중 나머지 23개 사건은 이 수집 범위에서 예측시장과 연결 검증을 마치지 못함. 현재 모델 입력과 채점은 공식 사건의 근거 판정이며, 해당 9개 계약의 YES/NO 예측이나 수익률을 평가한 것이 아님.

따라서 이 결과는 **‘실제 예측시장의 정보 가치/예측 실험’이 아니라 ‘공식 사건의 근거 유효성 진단’**임. 회의에서 논의한 ‘지금 어떤 정보가 더 유용한가?’라는 질문 전체를 검증한 것은 아님. 예측시장에 최종 확정된 결과나 이후 시장 snapshot은 모델 입력에 넣지 않음. 원문 자체의 사후 오염은 아래 별도 QC로 점검함.

- 공개 후보: [market_candidates.json](../data/temporal_evidence_poc_v1/market_candidates.json)
- 시간 조건을 확인한 연결: [market_links.json](../data/temporal_evidence_poc_v1/market_links.json)
- 공개 데이터 출처: [ForecastBench datasets](https://www.forecastbench.org/datasets/)

## 3. 두 번 실행한 이유

v1에서는 ‘현재’의 의미에 모호함이 있었음. 예를 들어 요약에 1월 29일 금리 결정이 정확히 남아 있어도, 모델이 ‘2월 19일까지 다른 일이 없었는지는 모른다’며 보류함. 이를 낡은 Memory의 오적용이나 금리 수치의 저장 실패로 세면 안 됨.

v2에서는 모든 조건에 동일하게 **‘현재 = 제공된 공식 기록에서 확인되는 최신 적용 상태’**라고 명확히 함. 원문·질문·정답·검색 결과와 **실제로 생성된 요약을 그대로 재사용**함. 틀린 사례만 재시도하지 않고 30개 × 3조건 모두 재평가함. 이 변화는 방법론 개선이 아니라 평가 범위 명확화임.

### 원문 QC와 v3 최종 집계

- **Amazon–iRobot:** 2024년 기사 본문 추출에 2026년 추천 기사 모듈이 섞였음을 발견함. 별도의 DOM 모듈만 제거하고 원래 기사 4개 본문 블록의 보존을 hash로 확인함(2,605 → 2,226자). 새 source ID와 경로로 저장하고, 해당 사례의 요약 생성 및 3개 평가 조건을 모두 재실행함. 질문·정답 의미는 바꾸지 않음.
- **VanEck:** 2024-01-11로 표기된 출시 PDF가 2024-04에 생성·수정된 파일이어서, 2024-01-12 cutoff 당시의 파일인지 확인할 수 없었음. 미래 정보가 있다는 확정 판정은 아니지만, 보수적으로 사례 전체를 최종 집계에서 제외함. 이 사례는 원래 모든 조건에서 정답이었으므로 오답을 제거한 선별이 아님.
- 수정하지 않은 28개는 v2 응답 그대로 유지하고, Amazon 수정본 1개만 합쳐 최종 29개로 집계함. 최초 원문·정답·v1/v2 결과는 모두 보존함.
- 다른 원문에서는 이번 검사로 cutoff 이후 본문을 발견하지 못했음. 다만 현재 웹에서 가져온 문서이며, 모든 파일이 해당 시점의 웹 snapshot과 동일함을 보증한 것은 아님.

상세: [원문 감사](temporal_evidence_source_audit.md), [최종 집계와 응답 참조](../result/temporal_evidence_poc_v3/reviewed_report.json).

## 4. 최종 결과 — QC 후 29개 사건·111문항

| 조건 | 근거 상태 판정 정확도 | 평균 전달 토큰 |
| --- | ---: | ---: |
| 원문 전체 | 100% · 111/111 | 9,733 |
| Naive retrieval Top-10 | 100% · 111/111 | 2,938 |
| 간단 요약 Memory | 93.7% · 104/111 | 551 |

토큰은 전달된 과거 기록 부분을 `o200k_base`로 센 것임. 공통 지시·질문과 실행 harness 토큰은 제외. 동일 토큰 예산 비교가 아니며 93.7%를 공식 Memory 방법론 성능으로 인용하면 안 됨. 전체 문서 묶음에서 일반 요약을 한 번 생성한 조건이지, 장기간 순차적으로 갱신하는 Memory 시스템의 평가가 아님.

### 개발 중 결과 — 원문 QC 전 30개·114문항

| 조건 | v1 | v2: 범위 명확화 | 평균 전달 토큰 |
| --- | ---: | ---: | ---: |
| 원문 전체 | 100% · 114/114 | 100% · 114/114 | 9,717 |
| Naive retrieval Top-10 | 100% · 114/114 | 100% · 114/114 | 2,950 |
| 간단 요약 Memory | 82.5% · 94/114 | 93.9% · 107/114 | 552 |

위 개발 결과는 수정 이력을 설명하기 위해 보존함. 원문 QC 후 결과와 분모를 섞지 않음.

문항 템플릿 자체에 강한 label 규칙이 있음. FOMC 4개 문항 중 과거 사실·2% 목표·미래 미확정의 세 범주는 전체 사례에서 정답 label이 고정되어 있음. 따라서 **전체 정확도보다 아래 핵심 문항 결과가 중요함.**

| 핵심 검사 · QC 후 최종 | 원문 | 검색 | 요약 |
| --- | ---: | ---: | ---: |
| 이미 효력을 잃은 전제를 현재에도 맞다고 수용한 오류 | 0/15 | 0/15 | 0/15 |
| FOMC 최신 금리 상태 판정 | 24/24 | 24/24 | 24/24 |
| 기업 사건의 15개 현재/과거 상태 판정 | 15/15 | 15/15 | 15/15 |
| 지속적인 2% 장기 목표를 확인 | 24/24 | 24/24 | 18/24 |

효력이 달라진 15개 전제는 FOMC 금리 변경 10개와 기업 사건 5개임. 세 조건을 합쳐 독립 표본 45개라고 해석하지 않음. 원문과 검색이 모두 만점이므로 첫 진단의 난도가 낮으며, 금융 일반의 오류율이나 실제 예측 업무의 품질로 확대 해석할 수 없음.

## 5. 실제 사례와 남은 오류

### 낡은 근거 오적용은 일어나지 않은 사례

JetBlue–Spirit의 과거 발표에는 합병을 계속 추진한다는 내용이 있음. 2024-03-04 발표에는 합병계약 종료가 명시됨. 다음날을 기준으로 평가한 세 조건 모두:

- ‘계약에 따른 합병을 여전히 추진 중’은 반박된 것으로 판정.
- ‘2023년에는 계속 추진하겠다고 발표했음’은 역사적 사실로 인정.
- ‘현재 계약 종료에 합의했음’은 지지되는 것으로 판정.

즉, 과거 사실을 지우지 않으면서 현재 상태의 변경을 구분함. [실제 요약 조건 응답](../result/temporal_evidence_poc_v2/corporate_jetblue_spirit/concise_summary/response.json)

### 최종 요약 조건의 오류 7개 — v2의 FOMC 응답과 동일

1. **6개: 지속적인 목표를 확인할 근거가 요약에서 명시적으로 사라짐.** 원문에는 2% 장기 물가목표가 있으나, 요약에는 ‘물가가 2%로 내려갈 전망’ 또는 ‘2%를 향한 진전’만 남음. 모델은 목표와 전망을 같은 뜻으로 단정하지 않고 `insufficient`로 보류함. 원문 기준 end-to-end 정보 손실이지만, 제공된 요약만 읽는 모델의 보류 자체는 합리적일 수 있음. 이미 무효화된 근거를 채택한 오류가 아님.
2. **1개: label–reason 불일치.** 2022-12-14의 금리 4.25–4.50% 기록은 요약에도 정확히 남아 있음. 그런데 응답은 label을 `contradicted`로 쓰고 이유에는 ‘Actually this matches the claim’이라고 적음. 정해둔 label 채점에서는 오답을 유지하되, 시간 맥락 손실로 분류하지 않음. [실제 응답](../result/temporal_evidence_poc_v2/fomc_20230201/concise_summary/response.json)

독립 assistant가 위 유형과 초기 성공 사례를 원문·저장 요약·응답으로 대조함. 사람 annotation 또는 독립 평가 모델의 대규모 검증을 했다는 뜻은 아님.

## 6. 지금 내릴 수 있는 결론

**단순한 상태 갱신에서는 모델이 낡은 정보와 현재 정보를 비교적 잘 구분함. 이번 POC는 원하는 핵심 실패의 증거가 아니라, 그 실패를 쉽게 전제해서는 안 된다는 결과임.**

- 확인됨: 간단한 요약은 ‘정책 목표’와 ‘수치 전망’처럼 서로 다른 역할의 정보를 다르게 보존할 수 있음.
- 확인되지 않음: Memory가 오래된 전제를 현재 근거로 적극 재사용한다는 현상, 정보의 현재 효용을 판단하지 못하는 일반적 한계, 이로 인한 실제 예측/금융 판단의 악화.
- 예측시장 결과는 사건 예측의 정답일 수 있지만, 어떤 과거 정보가 더 중요했는지의 정답을 자동으로 제공하지 않음.
- 이미 해결된 공개 사건을 사용하는 이상 모델의 사전 지식을 배제한 진정한 ex-ante forecasting이라고 주장할 수 없음.
- 사례를 일부러 복잡하게 바꾸거나 실패한 사례만 남기지 않음. 실제 예측시장 POC로 확장하려면, 미해결 사건에서 평가할 **정보 유용성의 기준**과 사건별 원문 확보를 먼저 확정해야 함. 현 단계 결과로 대규모 공식 Memory baseline을 시작할 근거는 아직 약함.

## 7. 실행 및 재현

- 모델 호출: 요약 생성 30회 + v1 최종 평가 90회 + v2 최종 평가 90회 + Amazon 원문 수정 후 요약 1회/평가 3회 = **214회**. 실행 오류 0.
- CLI 보고 총 input 3,018,044 tokens(cached 1,429,504 포함), output 67,206. 실행 harness 포함 수치로 위 전달 토큰과 다름. reasoning output 11,797은 별도 필드이며 output에 다시 더하지 않음. 비용 금액을 추정하지 않음.
- 초기 3사례는 실행 경로 점검용이며 확장 실행에서 동일한 prompt hash의 결과만 캐시 재사용함.
- 최초 원문 archive 68개 및 Amazon 본문 수정본 1개. 동일 문서가 여러 사건 history에 반복되며, 모두 독립 관측인 것은 아님.
- 오프라인 데이터/코드 및 최종 source-QC 집계 검사 **27개 통과**. model/network 호출 없이 source hash, 날짜, 입력 경계, 정답 매핑, 87개 최종 결과 참조 등을 검사함.
- 실행 경로는 official Codex non-interactive guidance를 확인해 구성하고, 기존 Luna low transport를 유지함. 재현 prompt·JSONL trace·source hash·정답·선택된 청크·저장 요약 모두 보존.
- 새 작업은 `replicate/`에만 저장. Git 초기화/원격 push 없음. 이전 Recall/Decision 결과는 변경하지 않음.

실행 명령:

```bash
cd /home/hob/workspace/alignment/replicate
.venv/bin/python method/temporal_evidence_sources.py
.venv/bin/python method/temporal_evidence_corporate_sources.py
.venv/bin/python method/temporal_evidence_eval.py --workers 8
.venv/bin/python method/temporal_evidence_eval.py --instruction-version v2 --workers 8
.venv/bin/python method/temporal_evidence_corporate_clean_copy.py
.venv/bin/python method/temporal_evidence_clean_review.py
.venv/bin/python -m unittest discover -s tests -p 'temporal_evidence_test.py' -v
```

실행 중 자료나 prompt를 변경하면 캐시가 거부함. 새 실험은 새 버전의 출력 경로에서 수행할 것.

- [v1 전체 결과](../result/temporal_evidence_poc_v1/report.json)
- [v2 전체 결과](../result/temporal_evidence_poc_v2/report.json)
- [QC 후 최종 결과](../result/temporal_evidence_poc_v3/reviewed_report.json)
- [QC 후 정답 및 근거 ID](../result/temporal_evidence_poc_v3/reviewed_gold.json)
- [실험 protocol](temporal_evidence_poc_protocol.md)
- [독립 설계·실행 감사](temporal_evidence_design_audit.md)
