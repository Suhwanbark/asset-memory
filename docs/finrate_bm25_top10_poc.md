# FIN-RATE BM25 top-10 PoC

최종 실행: `result/finrate_bm25_top10_gpt6_luna_v3/`.

기존 gold20과 동일한 LT-QA 10개, EC-QA 10개를 평가했다. 전체 43개 기업의 배포 원래 chunk 15,311개를 검색한다. 정답이나 gold ID를 검색 조건으로 사용하지 않는다.

## 검색 설정

- `bm25s==0.3.13`, Lucene scoring variant, `k1=1.2`, `b=0.75`.
- `text` 전체를 소문자화하고 `[a-z0-9]+`로 tokenize. 제목 추가, stemming, stopword 제거, 재분할, 본문 절단 없음.
- 질문 원문으로 검색하며 query rewrite, 기업·연도 필터, reranker 없음.
- 점수 내림차순, 동점이면 원래 chunk ID 오름차순. 상위 10개 서로 다른 원래 chunk의 본문 전체를 순위대로 제공.
- 검색 분석용 top-100 순위·점수·제목과 top-5/top-10 gold ID 일치 통계를 보존. Gold 정보는 검색 이후 분석에만 사용한다.
- 논문은 답변에 top-5를 제공하며 BM25+R 조건도 보고한다. 이번 top-10 BM25-only는 사용자 요청 설정이다. 논문이 공개하지 않은 analyzer/parameter를 동일하게 재현했다고 주장하지 않는다.

## 생성·채점

- 답변과 judge 모두 `gpt-6-luna`, reasoning `medium`. Worker pool 20, 답변 완료 직후 개별 judge.
- 기존 공식 answer prompt와 judge rubric을 재사용. Judge에는 원래 질문·reference answer·key_points·이번 답변만 제공한다.
- 최대 top-10 prompt가 약 620,703 o200k token으로 기존 실행기의 한도를 초과했다. 원문을 유지하기 위해 답변의 context 설정을 1,050,000, compaction threshold를 950,000으로 확장했다. 실제 app-server가 보고한 유효 context는 828,400이며 최장 문항의 실제 input은 628,568 token이었다.
- CLI `turn/start`의 합산 1,048,576문자 제한은 별개다. 공식 `thread/inject_items`로 전체 answer prompt를 user history에 넣고, 모든 답변에 동일한 짧은 user trigger `Answer the preceding question using its provided context.`를 사용한다. 원문 요약·절단·도구 호출은 없다.
- Judge는 기존 CLI 입력 방식과 272,000 context 설정을 유지한다. 단일 Luna judge이며 논문의 3-model ensemble이 아니다.
- 첫 성공한 최장 문항 답변은 그대로 재사용했다. 실패한 사전 transport 검사는 모델 호출 이전에 거부됐고 usage가 없다. 완료된 답변을 다시 생성하거나 좋은 점수만 선택하지 않았다.
- CPU only. 여유 공간이 10 GiB 미만이면 중단. 요청·전체 prompt·답변·judge·provider usage·raw events·코드 snapshot을 보존한다.

## 실행 및 검증

작업 경로: `/home/hob/workspace/asset/replicate`.

```bash
.venv/bin/python method/run_finrate_bm25_top10_gpt6_luna_v3.py
.venv/bin/python method/validate_finrate_bm25_top10_v3.py
```

실행기는 기존 동일 응답을 재사용하며 frozen artifact가 다르면 중단한다. 새 설정/표본 실험은 새 버전의 경로를 사용한다. v1은 길이 사전검사, v2는 CLI 한도 확인 기록이며 최종 점수는 v3에만 있다.

구현: `method/finrate_bm25_top10_v1.py`(색인·검색), `method/finrate_bm25_appserver_transport_v1.py`(전체 본문 전달), `method/run_finrate_bm25_top10_gpt6_luna_v3.py`(생성·즉시 채점).

## 결과

| 유형 | BM25 top10 CORRECT | 기존 gold-context CORRECT |
|---|---:|---:|
| LT-QA | 2/10 (20%) | 2/10 (20%) |
| EC-QA | 0/10 (0%) | 6/10 (60%) |
| 전체 | 2/20 (10%) | 8/20 (40%) |

BM25 top10: CORRECT 2, PARTIAL 7, INCORRECT 11, 실행 오류 0. 두 조건은 동일 질문·모델·medium·rubric을 사용하지만 근거 내용·입력 token 수와 답변 입력 packaging이 달라, 동일 token 예산 비교가 아니다.

Gold ID 기준으로 하나 이상 회수한 문항은 LT 7/10, EC 8/10이며, 모든 gold를 회수한 문항은 LT 1/10, EC 0/10이다. 질문별 gold 참조 94건 중 22건을 회수했다. 이 ID 일치 통계만으로 다른 chunk에 동등한 근거가 없는지까지 확정하지 않는다.

원자료: `report.json`, `report.txt`, `validation.json`, `paired_comparison.csv`, 각 q_id의 `retrieval.json`, `answer/`, `judge/`.

출처: [FIN-RATE v4 §4.1.2·Table 4](https://arxiv.org/html/2602.07294v4), [BM25S](https://github.com/xhluca/bm25s), [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Codex app-server](https://learn.chatgpt.com/docs/app-server).
