# 공동작업용 저장소 구조

연구 계획의 기준 파일은 루트 `plan.md`다.

| 경로 | 용도 |
|---|---|
| `configs/` | 모델, 검색 설정, 실행 예산, seed, 고정 질문 ID |
| `prompts/` | 답변, agent, bank writer, judge prompt |
| `method/corpus/` | 수집, 파싱, chunk 및 원문 catalog, SQLite |
| `method/retrieval/` | BM25, dense, hybrid |
| `method/agents/` | Agent 실행기와 chunk/raw 탐색 도구 |
| `method/bank/` | 기업별 memory 구축, 갱신, 과거 버전 조회 |
| `method/kg/` | Knowledge graph baseline |
| `method/evaluation/` | 표본 구성, 답변 생성, 채점, 지표와 trace 분석 |
| `sandbox/` | 격리된 Docker에서 web search agent로 질문 답변, trace·contamination 보존 (`sandbox/README.md`) |
| `tests/` | 파서, 도구, 평가 등 검증 코드 |
| `docs/` | 데이터 설명, 실험 설명, 결과 요약, 그림 |
| `data/` | 공시 원문, 배포 데이터, DB, 인덱스, bank; Git 제외 |
| `result/` | 실행별 답변, 채점, 전체 trace, 로그; Git 제외 |

분류 폴더는 향후 구현·정리를 위한 자리다. 기존 `method/*.py`와 검증된 실행 snapshot의 경로는 유지했다. 코드 이동은 import와 데이터 경로를 함께 정리하는 후속 작업이다. 아직 KG 및 모든 비교 방법이 구현됐다는 뜻은 아니다.

공동작업 시 큰 데이터와 전체 실행 결과는 공유 서버나 별도 저장소로 전달한다. 이 Git 저장소에는 코드, 계획, 설정과 결과 요약을 담는다. 현재 일부 기존 스크립트와 문서에는 이 서버의 절대경로 및 로컬 실행 환경에 대한 의존성이 남아 있다.

`context.md`, `process.md`와 과거 context archive는 서버 작업 이력으로 남기며 Git에서 제외한다. 연구 결정과 공동 검토 사항은 `plan.md` 또는 `docs/`에 기록한다.
