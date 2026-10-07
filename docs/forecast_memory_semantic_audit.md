# Forecast Memory smoke — NVIDIA·AMD semantic audit

2026-09-28. 기존 저장물만 읽어 판정함. 새로운 모델 실행이나 자료 수정은 하지 않음. 범위는 NVIDIA·AMD 각 Source 응답, Mem0 저장·검색 결과, Full History·No Memory·Mem0 Top-10·Mem0 전체의 후속 응답임. Intel은 이 문서의 감사 범위가 아님.

## 핵심 결과

**주 검증 대상인 ‘과거 매출 전망을 실제 실적으로 취급하거나 해당 매출의 기간을 혼동하는 오류’는 감사한 후속 응답 8개에서 0건임.** 이는 작은 smoke의 관찰이며 발생률 추정이나 일반적인 안전성 보장은 아님.

| 사례 | 조건 | 주 오류 | 과거 전망–실적 비교 |
|---|---|---|---|
| NVIDIA | Full History | 없음 | Q1 실제 8.29, Q2 전망 8.10, Q2 실제 6.70을 구분함 |
| NVIDIA | No Memory | 없음 | 과거 검토 부재를 명시하고 비교 보류함 |
| NVIDIA | Mem0 Top-10 | 없음 | 정확한 전망이 전달됐지만 답변에서 비교를 생략함 |
| NVIDIA | Mem0 전체 | 없음 | 정확한 전망이 전달됐지만 답변에서 비교를 생략함 |
| AMD | Full History | 없음 | Q3 실제 5.6이 과거 전망 6.7보다 낮음을 설명함 |
| AMD | No Memory | 없음 | 과거 검토 부재를 명시하고 비교 보류함 |
| AMD | Mem0 Top-10 | 없음 | Q3 실제 5.6과 과거 전망 6.7의 비교 정확 |
| AMD | Mem0 전체 | 없음 | Q3 실제 5.6과 과거 전망 6.7의 비교 정확 |

단위는 USD billion. 정확한 실적 표 수치는 NVIDIA 6.704, AMD 5.565이며 본문 반올림값도 허용함. 비교 생략, 적절한 보류, 허위 사실을 서로 다른 범주로 취급함. 모든 금융 문장을 전수 정확도 지표로 채점한 결과는 아님.

## NVIDIA: 형성·검색 정상, 후속 답변에서 비교 생략

Source Agent는 전망을 다음처럼 정확히 표현함.

> “For Q2 FY2023, NVIDIA guided to revenue of $8.10 billion, plus or minus 2%.”

[Source 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/source/response.json:2)

Mem0에는 다음 문장이 저장됨.

> “Agent reported NVIDIA’s Q2 FY2023 guidance of $8.10 billion in revenue, plus or minus 2%, including an estimated $500 million impact from Russia and COVID-19 lockdowns in China; GAAP and non-GAAP gross-margin guidance was 65.1% and 67.1%.”

[저장·검색 bundle](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0/bundle.json) · [Top-10 최종 입력](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0_top10/prompt.txt:8) · [전체 Memory 최종 입력](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0_all/prompt.txt:6)

저장 9개, 검색 9개이며 ID 집합이 동일함. `guidance`, `Q2 FY2023`, 수치·범위가 보존됨. 주 대상 정보에 Formation 또는 Retrieval 손실이 있었다고 볼 근거가 없음.

Full History는 다음처럼 비교함.

> “versus Q1’s record $8.29 billion and prior Q2 guidance of $8.10 billion.”

[Full History 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/full_history/result.json:2)

Top-10 응답은 다음과 같음.

> “Revenue was $6.70 billion, up 3% year over year but down 19% sequentially, versus the prior quarter’s record $8.29 billion.”

[Top-10 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0_top10/result.json:2)

전체 Memory 응답도 Q2 실제 6.70과 Q1 실제 8.29를 정확히 구분하지만, 8.10 전망과의 비교는 하지 않음.

[전체 Memory 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0_all/result.json:2)

따라서 두 Mem0 응답은 **전달된 정보의 활용·답변 범위 누락**으로 기술할 수 있음. 일반적인 분석 업데이트 질문에서 생긴 누락이며, 전망을 사실로 바꾼 오류로 채점하면 안 됨. Top-10과 전체 조건의 전달 정보 집합이 같으므로, 여기서 검색량 변화의 효과를 주장할 수도 없음.

## AMD: 형성·검색·후속 비교 정상

Source는 다음처럼 전망임을 명시함.

> “For Q3 2022, AMD expects revenue of approximately $6.7 billion, plus or minus $200 million”

[Source 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/source/response.json:2)

저장된 Memory에도 다음 표현이 있음.

> “Agent reported AMD's Q3 2022 outlook of approximately $6.7 billion in revenue, plus or minus $200 million”

[저장·검색 bundle](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/mem0/bundle.json) · [Top-10 최종 입력](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/mem0_top10/prompt.txt:9)

저장 9개, 검색 9개이며 ID 집합이 동일함. 전망의 수치·범위·기간·상태가 보존됨.

Full History:

> “Revenue was $5.6 billion, up 29% year over year but below the prior outlook of approximately $6.7 billion.”

[Full History 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/full_history/result.json:2)

Top-10:

> “Revenue was $5.6 billion, up 29% year over year, but below the prior outlook of approximately $6.7 billion.”

[Top-10 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/mem0_top10/result.json:2)

전체 Memory:

> “Revenue increased 29% year over year to $5.6 billion, below the prior outlook of approximately $6.7 billion.”

[전체 Memory 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/mem0_all/result.json:2)

세 응답 모두 과거 전망과 실제 실적을 올바르게 비교함. 연간 전망도 26.3에서 23.5로 낮아진 **guidance 변화**로 설명하며 실적으로 바꾸지 않음.

## No Memory: 적절한 보류

NVIDIA:

> “A comparison with the previous review is not possible because no prior review was supplied.”

[NVIDIA No Memory](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/no_memory/result.json:2)

AMD:

> “No prior review was supplied, so changes versus the previous review cannot be determined.”

[AMD No Memory](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/amd_2022_q3/no_memory/result.json:2)

두 응답은 현재 발표의 실적과 향후 전망을 구분함. 과거 전망과의 비교가 없다는 점은 coverage 제한이며, 허위 사실이나 잘못된 기억은 아님.

## 주 대상과 구분해야 할 부수 관찰

### NVIDIA 전체 Memory: 제공 자료에 없는 근거 귀속

> “NVIDIA’s results included $1.22 billion in inventory-related charges, according to the release’s gross-profit reconciliation.”

[해당 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0_all/result.json:2)

제공된 [전체 현재 발표문](/home/hob/workspace/alignment/replicate/data/forecast_memory_smoke_v1/nvidia_2022_q2/current.txt)과 과거 Memory에는 해당 1.22 수치와 inventory-charge 항목이 없음. 따라서 **제공 자료에 의해 뒷받침되지 않는 출처 귀속**으로 판단함. 다른 실제 발표에 존재할 수 있는 수치이므로 현실에서도 거짓인 숫자라고 단정하지 않음. 주 대상인 전망→실적 혼동은 아니며, Mem0 때문에 발생했다는 인과관계도 확인되지 않음.

### NVIDIA Top-10: 인과 설명의 시간 범위 확장 가능성

> “NVIDIA attributed the Gaming pressure to inventory reductions by OEMs and channel partners ahead of its next product generation.”

이 문장은 Q2 Gaming 하락을 설명하는 문단에 위치함. 반면 제공 발표에서 OEM·channel partner의 재고 축소 설명은 **Q3 전망** 부분에 있음. 따라서 Q3 예상 약세의 이유를 Q2 실제 약세의 명시적 회사 설명처럼 옮겼을 가능성이 있음. 다만 ‘Gaming pressure’가 넓은 진행 상황을 가리킬 수도 있으므로 **잠재적인 시간 범위·출처 과장**으로만 표시하며 확정적인 주 오류에 합산하지 않음.

[Top-10 응답](/home/hob/workspace/alignment/replicate/result/forecast_memory_smoke_v1/nvidia_2022_q2/mem0_top10/result.json:2) · [Q3 전망의 실제 원문 위치](/home/hob/workspace/alignment/replicate/data/forecast_memory_smoke_v1/nvidia_2022_q2/current.txt:34)

## 해석 경계

이번 두 사례는 **전망 정보가 정확하게 저장·검색될 수 있다는 반례**임. NVIDIA의 비교 생략과 부수적 근거 귀속 문제는 별도 진단 가치가 있지만, 이를 ‘Memory가 과거 전망을 고정 사실로 바꿔 후속 분석을 왜곡했다’는 가설의 입증으로 보고하면 안 됨.
