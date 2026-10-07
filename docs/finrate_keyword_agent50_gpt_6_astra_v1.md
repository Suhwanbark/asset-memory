# FIN-RATE 50문항 — Luna와 GPT-6 Astra 비교

- 질문: 동일 LT-QA 25 + EC-QA 25, 총50문항.
- Corpus: 전체43기업, 원래15,311개 chunk.
- 변경: 답변 모델. Judge는 GPT-6 Luna, reasoning effort는 medium 유지.
- 동일 조건: BM25 top-10 전체 본문·순서·prompt; agent search/grep/read·prompt·예산.
- 실행: worker20, 생성 직후 채점, CPU, 10GiB disk guard, 모든 provider-visible trace·usage 보존.

| 방법 | 답변 모델 | LT25 | EC25 | 전체50 |
|---|---|---|---|---|
| BM25 top-10 | GPT-6 Luna | 5/25 (20%) | 0/25 (0%) | 5/50 (10%) |
| BM25 top-10 | GPT-6 Astra | 4/25 (16%) | 4/25 (16%) | 8/50 (16%) |
| Naive agent | GPT-6 Luna | 3/25 (12%) | 2/25 (8%) | 5/50 (10%) |
| Naive agent | GPT-6 Astra | 8/25 (32%) | 11/25 (44%) | 19/50 (38%) |

## 판정 및 문항별 비교

정답률은 CORRECT만 정답으로 집계한다. PARTIAL·INCORRECT·FAILURE·실행 오류를 분모에서 제외하지 않는다.

- bm25: 새 모델만 정답6, Luna만 정답3, 둘 다 정답2. 새 판정: {'INCORRECT': 15, 'CORRECT': 8, 'PARTIAL': 26, 'FAILURE': 1}.
- naive_agent: 새 모델만 정답14, Luna만 정답0, 둘 다 정답5. 새 판정: {'PARTIAL': 17, 'CORRECT': 19, 'INCORRECT': 14}.

## 기존 Gold-context 결과 — 별도20문항

| 답변 모델 | LT10 | EC10 | 전체20 |
|---|---|---|---|
| GPT-6 Luna | 2/10 (20%) | 6/10 (60%) | 8/20 (40%) |
| GPT-5.6 Luna | 3/10 (30%) | 6/10 (60%) | 9/20 (45%) |

Judge는 둘 다 GPT-6 Luna. Gold chunk 본문만 제공하며 reference answer는 답변 모델에 주지 않았다. Gold50 추가 실행은 앞선 범위 정정 때 중지됐으며, 이번 모델 비교에서 gold 조건을 새로 실행하지 않았다.

## 해석과 검증

같은 질문·입력·judge 조건의 모델 비교이며, 방법 간 실제 토큰 소비가 동일하다는 뜻은 아니다. 단일 Luna judge를 사용하므로 judge의 한계는 남는다. 성적을 보고 질문을 교체하거나 정상 답변을 재생성하지 않았다.

- 상태: completed; 채점100/100.
- 검증: 100건의 모델·prompt·공식judge·판정 및 원문span3550개.
- 실행 오류: 0건. 원인과 trace는 validation/report에 보존.

[결과](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt_6_astra_judge6_v1/report.json) · [모델별 문항 비교](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt_6_astra_judge6_v1/model_comparison.csv) · [검증 및 사용량](/home/hob/workspace/asset/replicate/result/finrate_keyword_agent50_gpt_6_astra_judge6_v1/validation.json)
