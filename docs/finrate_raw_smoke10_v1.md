# Raw filing 저장 smoke — 10개 공시

- 상태: 저장·탐색 검증 완료. QA 모델 평가 아님.
- 위치: `/home/hob/workspace/asset/replicate/data/finrate_raw_smoke10_v1/`
- DB: `manifest.sqlite` — SQLite, 도구에서는 read-only로 접근.
- 범위: 7개 기업, 10개 공시의 primary HTML. 10-K 5개, 10-K/A 3개, 10-Q 1개, 8-K 1개.
- 선택: 기존 확보본 12개를 filing key로 정렬 → 기업별 1개 → 빠진 공시 유형 → 정렬 순서로 10개까지 채움. 저장 smoke용이며 무작위 평가 표본이 아님. 질문·gold·모델 점수를 사용하지 않음.
- 추가 다운로드·LLM 호출·GPU 사용: 0. 10 GiB 여유 공간 중지 기준 적용.
- 현재 저장 정책: 사용자 요청으로 `structure.json` 10개와 DB의 `structure_path` 열을 제거했다. 이후 builder도 해당 파일을 저장하지 않는다. Schema version 2. 원본 HTML·TXT와 기존 탐색 출력은 동일하다.

## 저장 구조

```text
finrate_raw_smoke10_v1/
├── manifest.sqlite
├── selection.json
└── filings/<CIK>/<accession-without-hyphens>/
    ├── primary.html       # 기존 공개 HTML 사본의 바이트 그대로
    ├── primary.txt        # 전체 검색·열람용 본문; 고정 길이 chunking 없음
    └── provenance.json    # 공식 metadata, 실제 취득 출처, 변환 설정
```

본문은 숫자·구두점에 Markdown escaping을 추가하지 않는 UTF-8 text로 저장했다. HTML의 문단·행 구분과 셀 사이 tab을 유지한다. 셀 내부 문단에도 줄바꿈이 있을 수 있어 평면 TSV로 간주하지 않는다. 원래 표 구조는 보존된 HTML을 필요할 때 다시 파싱해 확인한다. `structure.json`은 저장하지 않는다.

| SQLite table | 레코드 | 의미 |
|---|---:|---|
| `filings` | 10 | accession, CIK, ticker, 공식 form, 제출일·보고일, 정정 여부 |
| `documents` | 10 | 공시별 primary 문서, 출처 URL, 파일 경로·hash |
| `sections` | 1,506 | 문서 root 10, 원문 TOC 기반 Item/서명 탐색 주소 134, HTML table 1,362 |
| `text_spans` | 62,487 | 본문 문자 구간 ↔ 정제하기 전 HTML DOM 주소 |
| `finrate_mapping` | 287 | 기존 chunk ID의 공시 후보 연결. chunk 내부 원문 offset은 미확정(NULL) |

HTML table 수에는 레이아웃용 표도 포함된다. Item 주소 수는 전체 의미적 섹션 수나 본문 보존율이 아니다. 섹션 index에 없는 본문도 문서 전체에 남아 있어 `grep`/`read`로 접근한다. 현재 parent는 문서 root이며 의미적 세부 계층은 추정하지 않았다.

## 저장된 공시

| 기업 | 공식 form | 공식 제출일 | SEC reportDate | accession |
|---|---|---|---|---|
| GEN | 10-K | 2023-05-25 | 2023-03-31 | 0000849399-23-000014 |
| GS | 10-K/A | 2024-02-28 | 2023-12-31 | 0000886982-24-000012 |
| GS | 10-K | 2025-02-27 | 2024-12-31 | 0000886982-25-000005 |
| HUSA | 10-K | 2022-03-31 | 2021-12-31 | 0001493152-22-008596 |
| HUSA | 10-K/A | 2025-04-30 | 2024-12-31 | 0001641172-25-007857 |
| LANC | 10-K | 2024-08-22 | 2024-06-30 | 0000057515-24-000020 |
| NXPL | 10-K | 2023-03-31 | 2022-12-31 | 0001493152-23-010400 |
| REEMF | 10-K/A | 2022-04-27 | 2021-12-31 | 0001419806-22-000010 |
| REEMF | 8-K | 2024-10-25 | 2024-10-25 | 0001419806-24-000023 |
| VLO | 10-Q | 2024-10-30 | 2024-09-30 | 0001035002-24-000056 |

SEC `reportDate`는 8-K에서는 사건 보고일일 수 있어 모든 문서에 회계기간 종료일이라는 의미를 부여하지 않았다. 문서에 포함된 개별 사실의 대상 기간은 별도 문제다. 기업명 필드는 현재 확보된 SEC metadata snapshot의 명칭이며, 과거 ticker의 회사명을 소급 추정하지 않았다. 정정 공시는 별도 accession으로 보존했고, 정정 대상 원공시는 확인 전까지 NULL로 둔다.

## 탐색 예시

프로젝트 경로에서 실행:

```bash
python method/finrate_raw_store_v1.py list_filings '{"ticker":"HUSA","include_sections":true}'
python method/finrate_raw_store_v1.py grep '{"pattern":"Item 2.01","limit":5}'
python method/finrate_raw_store_v1.py read '{"document_id":"0001419806-24-000023:primary","start":0,"length":2000}'
```

- `list_filings`: ticker/form/제출일 필터, offset pagination, 선택적 Item 목록.
- `grep`: 전체 또는 지정 문서의 literal 검색, 대소문자 옵션, 위치·주변 본문, pagination. 이 저장 smoke에서는 regex 미지원.
- `read`: 문서·문자 위치 또는 section ID로 열람. `next_start`로 이어 읽기. 최대 24,000문자/호출.
- 주소: Unicode 문자 기준, 시작 포함·끝 제외. 각 read는 문서 hash와 원문 DOM 주소를 반환.
- 이 도구 검증은 프로그램이 수행했다. Agent의 탐색 workflow나 답변 정책은 만들지 않았다.

## 검증 결과

- 10/10 HTML 파일이 기존 취득본 SHA-256과 동일.
- 10/10 accession·form·제출일·reportDate·primaryDocument를 보존된 SEC submissions JSON과 재대조.
- 62,487개 text span이 **정제 전** 원본 DOM의 문자열과 일치.
- 10/10 문서를 여러 read로 끝까지 읽어 결합한 결과가 저장된 전체 본문과 동일.
- 목록 필터·pagination, 전체/문서별 검색·pagination, Item 이동 검증 통과.
- SQLite integrity/foreign key 검증 통과.
- 숨김 노드 제거 후 원문 주소, 표 헤더·음수 괄호·셀 span, 목차→본문 이동 regression 3개 통과.
- 도구 호출·정확한 출력 150건: `result/finrate_raw_smoke10_v1/tool_trace.jsonl`.
- 초기 데이터 총 65,346,138 bytes(약 62.32 MiB). 이 수치와 당시 구현은 초기 smoke 이력으로 보존한다.
- `structure.json` 제거 후 데이터: 약 **40.22 MiB**. 구조 JSON 파일 **22.10 MiB** 제거.
- 제거 후 HTML·TXT hash 10건 일치, 기존 도구 trace **150호출의 출력이 모두 동일**. DB integrity·foreign key 및 새 builder schema 일치 검증 통과.
- 초기 보고서: `result/finrate_raw_smoke10_v1/report.json`.
- 제거·검증 보고서: `result/finrate_raw_smoke10_v1/structure_removal_report.json`.
- 초기 구현 snapshot: `result/finrate_raw_smoke10_v1/source_snapshot/`.
- 현재 구현 snapshot: `result/finrate_raw_smoke10_v1/source_snapshot_after_structure_removal/`.

## 전체 공시 용량 추정

대상은 FIN-RATE 제목 기준 **2,472개 공시 묶음**이다. SEC accession 전수 대조가 끝난 고유 공시 수는 아니다. Primary HTML만 수집하며, 외부 이미지·별첨·추가 검색 색인·임베딩·bank·실행 trace는 아래 추정에서 제외한다.

| 구성 | 본문 분량으로 보정한 추정 |
|---|---:|
| HTML 원본 사본 | 약 1.39 GiB |
| 전체 열람용 TXT | 약 0.17 GiB |
| 현재 구조의 SQLite | 약 0.91 GiB |
| 합계 | 약 **2.48 GiB** |

- 전체 배포 corpus의 본문 UTF-8 bytes: 235,569,836 bytes(224.66 MiB).
- Smoke 10건에 대응하는 배포 본문: 3,730,739 bytes. 각 저장 구성의 실제 bytes에 전체/표본 본문 분량 비율을 곱했다.
- 확보본 12건 전체를 사용한 HTML 교차 추정은 1.36 GiB로 비슷하다.
- 공시 개수로 단순 환산하면 약 9.71 GiB지만, smoke의 8/10이 연차보고서·정정본이고 전체는 제목 기준 161/2,472가 10-K 묶음이어서 문서 길이 편중이 크다.
- 분량 보정도 정확한 예측이나 상한은 아니다. 배포 corpus의 원문 누락, 문서 유형별 HTML 스타일·표 비율, 원문 매핑 결과에 따라 달라질 수 있다. DB 크기의 본문 비례 가정도 근사다.
- 운영 공간은 우선 **4–5 GiB를 배정하는 제안**이며 통계적 신뢰구간이나 보장된 상한이 아니다. 서버 여유 10 GiB 미만 중지 기준은 유지한다.
- 계산 자료: `result/finrate_raw_smoke10_v1/capacity_estimate.json`. 전체 다운로드는 아직 시작하지 않았다.

## 출처와 현재 범위

SEC accession metadata에 연결된 **Yahoo Finance 공개 HTML 사본**을 재사용했다. 공식 SEC URL과 실제 다운로드 URL을 분리해 저장하며, SEC가 직접 반환한 파일과의 byte 일치는 검증하지 않았다. primary HTML 자체는 통째로 보존한다. 외부 이미지·별첨은 다운로드하지 않았으며 참조 URL만 남아 있다. 따라서 완전한 제출 패키지나 PDF/브라우저 시각 재현 환경은 아니다.

본문 projection은 head/script/style/inline-XBRL hidden 및 명시적으로 숨김 처리된 요소를 제외하고 공백을 정규화한다. 특정 Item이나 질문 관련 내용만 선별하지 않는다. 표의 시각적 정렬·이미지 의미까지 검증한 것은 아니므로 향후 QA 전에 표 열람 인터페이스를 점검해야 한다.

기존 FIN-RATE 묶음과 연결한 287개 chunk는 문서 후보 metadata 연결까지다. 원문을 기존 chunk를 이어 붙여 생성하지 않았고, 기존 chunk의 정확한 원문 span이나 모든 내용이 이 primary 문서에 포함된다는 판정을 만들지 않았다. 이전 QA 결과·고정 질문·배포 corpus는 수정하지 않았다.
