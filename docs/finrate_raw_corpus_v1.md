# FIN-RATE raw filing corpus

## 2026-10-07 21:23 KST 실행 결과

| 문서 묶음 상태 | 수 |
|---|---:|
| 검색 대상으로 매핑 채택 | 1,342 / 2,472 (54.29%) |
| 사본은 있으나 매핑 모호 | 133 |
| 사본 미확보 | 997 |

선택된 공시 파일은 1,348개다. 12,096개 후보 원문 파일을 확보·변환했으며 대안 후보는 별도로 표시하고 기본 agent 검색에서 제외한다. 전체 저장량 약5.46GiB에는 후보 파일도 포함하며, 선택된 원문+TXT 본문 합은 약0.92GiB다. 12,096개 파일 쌍의 해시와 DB integrity/foreign keys,43기업의 도구129호출·전체read roundtrip43건을 검증했다. 오류0,미완료part파일0,수집프로세스종료.

**2020년389개와2021년473개 묶음은 이번 공개사본 경로에서 모두 미확보다.** SEC metadata는 확보했지만 HTML 직접 다운로드는403이었다. 따라서 FIN-RATE 전체와 동일한 raw filing 실험 환경이 완성됐다고 해석하면 안 된다. 전체 원문을 갖추려면 미확보 원문을 접근 가능한 출처에서 보충하고133개 모호한 매핑을 검토해야 한다. QA 평가나모델호출은 수행하지 않았다.

- 데이터: `/home/hob/workspace/asset/replicate/data/finrate_raw_v1/`
- 진행 상태: `result/finrate_raw_ingestion_v1/status.json`
- 완료 보고서: `result/finrate_raw_ingestion_v1/report.json` (생성 후 사용)
- 기업별 coverage: `result/finrate_raw_ingestion_v1/company_coverage.json`
- 문서 묶음별 성공·미확보·모호성: `result/finrate_raw_ingestion_v1/group_coverage.json`

## 범위와 저장 구조

FIN-RATE 배포 corpus의 43개 기업, 2,472개 `ticker + title date + form` 묶음 전체를 대상으로 한다. 이 묶음은 SEC accession과 일대일이 아니다. 특히 같은 날 같은 서식으로 제출된 다수 증권공시가 있어 SEC metadata 후보는 21,035개다. 후보 전체를 실제 FIN-RATE 공시 수라고 집계하지 않는다.

```text
finrate_raw_v1/
├── manifest.sqlite
├── ingestion_plan.json
├── metadata/
│   ├── company_cik_candidates.json
│   └── sec_submissions/*.json
└── filings/<CIK>/<accession without hyphens>/
    ├── primary.htm      # 취득한 원본 바이트; 일부 공시는 primary.txt
    └── primary.txt      # HTML의 전체 읽기용 본문
                        # 원본이 TXT이면 읽기용은 readable.txt
```

기업명·CIK·실제 공시 유형·공시일·접수시각·보고일, 원문 및 취득 URL, 파일 경로·해시는 SQLite에 저장한다. `structure.json`, 개별 `provenance.json`, DOM text-span DB는 생성하지 않는다. 원본 HTML의 특정 Item을 선택하거나 본문을 고정 길이 chunk로 나누지 않는다. 읽기용 텍스트는 script/style/숨김 요소를 제외하며 표의 행·셀 구분을 남긴다. 시각적 표 렌더링을 완전히 재현한다는 뜻은 아니다. 원본 바이트는 보존한다.

수집 단위는 공시의 **primary document**다. 별도 exhibit 파일이나 외부 이미지까지 수집하는 전체 제출 패키지는 아니다.

## SQLite

| 테이블 | 역할 |
|---|---|
| `filings` | SEC 공시 metadata와 FIN-RATE 기업 alias |
| `documents` | 원문/TXT 경로, 취득 상태·오류, URL·해시, corpus 선택 상태 |
| `title_groups` | 2,472개 원래 문서 묶음과 상태 |
| `group_candidates` | 원래 묶음과 SEC 공시 후보의 다대다 연결 |
| `finrate_mapping` | 15,311개 기존 chunk의 원문 후보 매핑과 검증 수준 |
| `content_checks` | 후보별 결정적 본문 probe 일치 수 |
| `document_selection` | corpus에 선택된 문서와 선택 근거 |
| `sections` | 전체 본문 주소와 Item 줄의 경량 탐색 힌트 |
| `acquisition_attempts` | URL별 실제 다운로드 결과·오류 |
| `store_metadata` | 파서·수집 범위·설정 |

`corpus_documents` view는 선택된 원문만 반환한다. 내려받았지만 다른 공시일 가능성이 있는 후보는 `candidate_not_selected`로 구분한다.

## 매핑과 취득의 한계

공식 SEC 제출 이력에서 날짜와 서식을 대조한다. FIN-RATE 제목 날짜가 SEC의 공식 filing date와 다르면 미국 동부 기준 접수일도 확인한다. ±3일 후보는 따로 표시하고 본문 확인을 요구한다. CIK는 기업명·ticker 변경과 분리하여 유지한다.

단일 날짜/서식 후보 또는 충분한 본문 probe 일치가 있는 후보를 선택한다. Probe는 각 기존 chunk에서 결정적으로 추출하며 질문·정답·성능을 사용하지 않는다. 본문 일치율은 identity 확인의 보조 지표이고, 모든 Item·표·기존 chunk가 완전히 보존됐다는 증거는 아니다. 기존 chunk의 원문 exact offset을 추정해서 채우지 않는다.

이 서버의 SEC HTML 직접 요청은 403으로 실패하여, SEC metadata와 accession/filename이 대응하는 공개 Yahoo Finance 사본을 수집한다. 취득 URL과 SEC 원문 URL을 구분하며 SEC 원본과의 byte 대조는 미완료다. 404, 원문 미확보, 후보 모호성을 최종 coverage에 남긴다.

## Agent 인터페이스

```bash
python method/finrate_raw_store_v2.py list_filings --args '{"ticker":"GS","limit":20}'
python method/finrate_raw_store_v2.py grep --args '{"query":"revenue","ticker":"GS","limit":20}'
python method/finrate_raw_store_v2.py read --args '{"document_id":"실제 document_id","start":0,"length":12000}'
```

- 기본 범위는 매핑 후 선택된 원문이다. 매핑 전에는 QA corpus로 공개하지 않는다.
- `grep`: literal, case-insensitive 검색; 문서 ID와 정확한 문자열 offset 반환.
- `read`: 문서 전체를 주소화한 문자 단위 창, 다음 위치 `next_start` 제공; 최대 24,000자.
- `list_filings`: 기업 alias·실제 SEC 서식 필터와 pagination.
- 도구는 탐색 순서나 기업·연도 비교 workflow를 강제하지 않는다.

## 실행 및 검증

48개 다운로드 worker, 공개 CDN 요청 시작 최대 초당 20건, CPU 변환 worker 8개. SEC metadata 수집은 별도로 최대 초당 6.67건으로 수행했다. 여유 공간 10GiB 미만이면 중지한다. QA 답변·judge·embedding·GPU 실행은 없다.

획득 파일/TXT 해시와 SQLite 무결성·외래키를 검증한다. 선택 원문이 있는 각 기업에 대해 목록·검색·읽기와 전체 본문 read roundtrip을 검사하고 실제 반환 trace를 보존한다. 현재 진행률·완료 여부는 위 JSON 기록이 기준이다.

전체정규화chunk대조추가: GS_2022-08-22_424B2의doc_010524(56,844정규화문자)가단일후보에전체포함됨을확인해1개묶음을추가연결했다. 최초16probe동률30묶음대상의감사이며, 원문파일은변경하지않았다. 초기DB와보고서는 before_full_chunk_refinement/에보존했다.
