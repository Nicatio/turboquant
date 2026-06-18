# TurboQuant 작업 히스토리 총정리

작성일: 2026-05-05  
대상 저장소: `Turboquant`

## 1. 문서 목적

이 문서는 현재 저장소에서 지금까지 진행된 TurboQuant 관련 작업을 한 번에 파악할 수 있도록 정리한 히스토리 문서다.  
단순 커밋 로그 나열이 아니라, 아래 항목을 함께 묶어 정리한다.

- 커밋 기준 기능 진화 과정
- 커밋되지 않았지만 실제로 추가/검증된 확장 작업
- 주요 벤치마크와 관찰 결과
- 현재 저장소 상태
- 남아 있는 이슈와 다음 과제

---

## 2. 한 줄 요약

이 저장소는 처음에는 TurboQuant 논문 구현체로 시작했지만, 현재는 다음까지 포함하는 **Apple Silicon 기반 MLX 실험/서빙 스택**으로 확장되었다.

- TurboQuant 논문 핵심 알고리즘 구현
- MLX 기반 KV-cache 압축 실험
- `shadow` / `direct` 두 가지 캐시 경로
- Llama / Qwen / Gemma 4 대상 실모델 검증
- Needle / LongBench / GloVe / Wikitext-2 / `turboquant_plus` 비교 벤치마크
- 로컬 OpenAI-compatible Gemma 4 서버
- Continue 연동용 설정

---

## 3. 커밋 기준 타임라인

### 3.1 `0cb6fbe` Initial commit

초기 저장소 생성 커밋.

### 3.2 `6ca675e` `feat: implement turboquant`

TurboQuant의 첫 실질 구현 단계.

핵심 작업:

- 패키지 구조 및 `pyproject.toml` 추가
- 논문 핵심 구성요소 구현
  - `src/turboquant/mse_quantizer.py`
  - `src/turboquant/prod_quantizer.py`
  - `src/turboquant/qjl.py`
  - `src/turboquant/rotation.py`
  - `src/turboquant/lloyd_max.py`
- 기본 KV-cache 경로 구현
  - `src/turboquant/kv_cache.py`
- 평가/실험 스크립트 추가
  - `scripts/run_synthetic_mse_eval.py`
  - `scripts/run_inner_product_eval.py`
  - `scripts/run_nn_benchmark.py`
  - `scripts/eval_mlx_input_embedding_accuracy.py`
  - `scripts/eval_mlx_turboquant_kv_memory.py`
  - `scripts/run_mlx_smoke.py`
  - `scripts/run_mlx_turboquant_prompt.py`
- 기초 테스트 세트 추가
  - `tests/test_mse_quantizer.py`
  - `tests/test_prod_quantizer.py`
  - `tests/test_qjl.py`
  - `tests/test_rotation.py`
  - `tests/test_lloyd_max.py`
  - `tests/test_kv_cache.py`

의미:

- 논문 수식 기반 reference implementation이 갖춰진 시점
- 로컬 CPU/MLX 검증용 골격이 완성됨

### 3.3 `4c8d371` `doc: add readme`

문서화 단계.

핵심 작업:

- `README.md` 추가

의미:

- 저장소 목적, 사용법, 실험 방향을 정리하는 외부 진입점이 생김

### 3.4 `64d9b89` `feat: mlx support on kv cache`

MLX 기반 캐시 실험이 본격화된 단계.

핵심 작업:

- MLX 양자화 구현 추가
  - `src/turboquant/mlx_quantizer.py`
- KV-cache 로직 확장
  - `src/turboquant/kv_cache.py`
- 벤치마크/평가 유틸 추가
  - `src/turboquant/benchmark_utils.py`
- 공용 벤치마크 스크립트 추가
  - `scripts/run_mlx_needle_benchmark.py`
  - `scripts/run_longbench_e_mlx.py`
  - `scripts/run_glove_benchmark.py`
- 관련 테스트 추가
  - `tests/test_benchmark_utils.py`

의미:

- 단순 수학 구현에서 벗어나 실제 MLX 추론 경로와 가까운 실험이 가능해짐
- LongBench / Needle / GloVe 등 공개 벤치마크 축이 들어옴

### 3.5 `b1c4363` `feat: add a newer direct TurboQuant KV path`

현재 구조의 큰 전환점.

핵심 작업:

- `direct` KV-cache 경로 추가/확장
  - `src/turboquant/kv_cache.py`
- Metal 보조 커널 추가
  - `src/turboquant/metal_kernels.py`
- 모델 attention 래퍼 추가
  - `src/turboquant/mlx_attention.py`
- VLM/멀티모달 경로 추가
  - `src/turboquant/mlx_vlm_utils.py`
- Qwen VLM 벤치마크 추가
  - `scripts/eval_mlx_vlm_turboquant_kv.py`
  - `scripts/run_mlx_vlm_smoke.py`
- 샘플 이미지 추가
  - `assets/sample_grid.ppm`
- direct 경로 전용 테스트 추가
  - `tests/test_direct_kv_cache.py`
  - `tests/test_mlx_attention.py`

의미:

- 이 시점부터 저장소가 “reference 구현”을 넘어 “실모델 KV-cache 연구 플랫폼”에 가까워짐
- `shadow` 와 `direct` 의 양대 경로가 자리잡음

---

## 4. 커밋 이후 확장 작업 (현재 작업트리에 반영됨, 일부 미커밋)

아래 항목들은 현재 저장소에 반영되어 있으나, 위 커밋 로그에는 모두 반영되지 않은 작업들이다.

### 4.1 Qwen 확장 및 비교 벤치마크

추가된 스크립트/결과:

- `scripts/eval_qwen_wikitext2_ppl.py`
- `scripts/run_qwen_mode_benchmark.py`
- `scripts/run_qwen_vlm_turboquant_sweep.py`
- `scripts/compare_turboquant_plus.py`
- `reports/benchmarks/qwen_mode_sweep.csv`
- `reports/benchmarks/qwen_vs_turboquant_plus.csv`
- `reports/benchmarks/qwen_wikitext2_ppl.csv`

핵심 작업:

- Qwen multimodal 경로에서 TurboQuant KV-cache 평가
- Wikitext-2 기반 PPL 비교
- `turboquant_plus`와의 구조적/수치적 비교
- 모드별(`shadow`, `direct`, adaptive 조합) 스윕 자동화

의미:

- “우리 구현이 실제로 어디까지 괜찮은가”를 외부 구현체와 공개 지표로 비교할 수 있게 됨

### 4.2 Gemma 4 지원

추가된 스크립트:

- `scripts/eval_mlx_gemma4_turboquant_kv.py`
- `scripts/run_gemma4_long_context_throughput.py`

관련 코드:

- `src/turboquant/mlx_attention.py`
- `src/turboquant/kv_cache.py`
- `src/turboquant/mlx_quantizer.py`

핵심 작업:

- `mlx-community/gemma-4-26b-a4b-it-4bit` 기준 KV-cache 실험
- Gemma 4의 hybrid attention 구조에 맞춘 캐시 적용
- 장문맥(`128k`, `256k`) throughput 측정용 벤치마크 추가

의미:

- Qwen뿐 아니라 Gemma 4까지 실모델 커버리지가 확장됨
- 장문맥 실험의 기준 모델이 하나 더 생김

### 4.3 OpenAI-compatible 로컬 서버

추가된 파일:

- `src/turboquant/openai_compatible_server.py`
- `scripts/serve_openai_gemma4.py`
- `tests/test_openai_compatible_server.py`

핵심 작업:

- Gemma 4를 OpenAI-style API로 서빙
- `/v1/chat/completions` 스타일 인터페이스 제공
- tool call 파싱/정규화 지원
- 내부 thought 채널 제거 처리

의미:

- 단순 벤치마크 저장소를 넘어, 편집기/에이전트에서 쓸 수 있는 로컬 모델 서버 역할까지 수행 가능

### 4.4 Continue 연동 보조 파일

현재 저장소 내 파일:

- `.continue/config.yaml`
- `.continue/rules/tool-use-fallback.md`

핵심 작업:

- Continue에서 로컬 Gemma 4 서버를 모델로 붙이기 위한 설정
- raw tool-call markup을 그대로 출력하지 않도록 방어 규칙 추가

주의:

- Continue 연동 과정에서 홈 디렉터리 쪽 설정과 충돌/조정이 있었고, 저장소 안 `.continue`는 그 일부만 남아 있는 상태다.

### 4.5 장문맥 벤치마크 러너 확장

핵심 파일:

- `scripts/run_gemma4_long_context_throughput.py`

핵심 작업:

- `baseline-only`, `turbo-only`, `both` 실행 모드 추가
- `2k`, `128k`, `256k` 프롬프트 길이에서 prompt/decode throughput 측정
- 동일 프로세스 연속 실행으로 인한 메모리 잔존 영향과 분리하기 위해 isolated run 지원

의미:

- “같은 프로세스에서 baseline 후 TurboQuant를 돌려서 생기는 왜곡”을 줄이고 더 공정한 측정이 가능해짐

---

## 5. 현재 아키텍처 요약

### 5.1 논문 구현 레이어

핵심 파일:

- `src/turboquant/mse_quantizer.py`
- `src/turboquant/prod_quantizer.py`
- `src/turboquant/qjl.py`
- `src/turboquant/rotation.py`
- `src/turboquant/lloyd_max.py`

역할:

- TurboQuant의 수학적 핵심 구현
- MSE 기반 벡터 양자화
- inner-product preserving 추정
- Lloyd-Max 코드북 계산
- 랜덤 회전 및 QJL 보정

### 5.2 MLX 양자화/캐시 레이어

핵심 파일:

- `src/turboquant/mlx_quantizer.py`
- `src/turboquant/kv_cache.py`
- `src/turboquant/metal_kernels.py`

역할:

- MLX에서 돌아가는 양자화/복원 경로
- `shadow` / `direct` 캐시 구현
- mixed-bit, shared quantizer pool, recent window/slack 등 실용 기능
- Metal 보조 커널 제공

### 5.3 모델 통합 레이어

핵심 파일:

- `src/turboquant/mlx_attention.py`
- `src/turboquant/mlx_vlm_utils.py`

역할:

- Llama/Qwen/Gemma 4 attention 래핑
- 모델별 KV-cache 경로 주입
- VLM/멀티모달 모델 로딩 보조

### 5.4 벤치마크/평가 레이어

핵심 파일:

- `src/turboquant/benchmark_utils.py`
- `src/turboquant/metrics.py`
- `src/turboquant/nn_eval.py`

및 주요 스크립트:

- `scripts/run_mlx_needle_benchmark.py`
- `scripts/run_longbench_e_mlx.py`
- `scripts/run_glove_benchmark.py`
- `scripts/eval_mlx_vlm_turboquant_kv.py`
- `scripts/eval_qwen_wikitext2_ppl.py`
- `scripts/run_qwen_mode_benchmark.py`
- `scripts/run_gemma4_long_context_throughput.py`

### 5.5 서빙 레이어

핵심 파일:

- `src/turboquant/openai_compatible_server.py`
- `scripts/serve_openai_gemma4.py`

역할:

- 로컬 MLX Gemma 4를 OpenAI-compatible server 형태로 노출
- Continue 같은 외부 툴과 연결 가능한 인터페이스 제공

---

## 6. 지금까지 확인된 주요 벤치마크/관찰 결과

### 6.1 테스트 상태

현재 로컬 테스트 결과:

- `48/48` 통과

실행 명령:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

### 6.2 Gemma 4 baseline short-context 성능

최근 재검증 기준:

- 모델: `mlx-community/gemma-4-26b-a4b-it-4bit`
- 프롬프트 길이: 약 `2046` 토큰
- 생성 길이: `32` 토큰

결과:

- prompt throughput: `589.072 tok/s`
- generation throughput: `55.402 tok/s`
- peak memory: `15.204 GB`

해석:

- 짧은 컨텍스트에서는 Gemma 4 baseline이 꽤 빠르며, “로컬 Gemma 4는 50~60 tok/s”라는 체감과 비슷한 수준이다.

### 6.3 Gemma 4 long-context baseline vs shadow (`256k`)

isolated run 기준:

#### Baseline-only

- prompt throughput: `198.013 tok/s`
- generation throughput: `19.400 tok/s`
- peak memory: `23.390 GB`
- cache storage: `5.457838 GB`

#### TurboQuant shadow-only (`3.5-bit`, `dense_shadow`)

- prompt throughput: `190.560 tok/s`
- generation throughput: `16.258 tok/s`
- peak memory: `25.969 GB`
- cache storage: `1.409096 GB`
- storage compression: 약 `3.87x`

해석:

- 장문맥에서는 baseline 자체도 decode가 많이 느려진다.
- `shadow + dense_shadow` 는 cache storage는 크게 줄이지만, 현재 구현에서는 runtime peak memory가 오히려 더 높다.
- 이유는 dense reconstructed KV와 index shadow를 같이 들고 있기 때문이다.

### 6.4 Gemma 4 direct 경로 상태

현재 상태:

- 장문맥 benchmark 용으로는 **정상 사용 불가**

관찰:

- `256k` direct-only 실패
- 약 `2046` 토큰 수준에서도 실패

대표 에러:

```text
ValueError: [broadcast_shapes] Shapes (256,256) and (1,16,256,1279) cannot be broadcast.
```

해석:

- Gemma 4의 shared KV / sliding attention / direct cache 경로 사이에 mask shape 정합성 문제가 남아 있다.
- 따라서 현재 Gemma 4에서는 `direct`는 연구/수정 대상이고, 실사용 benchmark 경로는 `shadow` 쪽이 더 안정적이다.

### 6.5 Qwen 관련 축

현재 결과물:

- `reports/benchmarks/qwen_mode_sweep.csv`
- `reports/benchmarks/qwen_vs_turboquant_plus.csv`
- `reports/benchmarks/qwen_wikitext2_ppl.csv`

요지:

- `shadow`, `direct`, adaptive tail 조합을 수치로 비교할 수 있음
- `turboquant_plus`와의 비교를 CSV로 남겨둠
- Qwen multimodal 경로 및 PPL 축까지 들어와 있음

### 6.6 공개 벤치마크 축

지원/추가된 축:

- Needle in a Haystack
- LongBench(-E) adapter
- GloVe 기반 ANN benchmark
- Wikitext-2 PPL
- `turboquant_plus` 비교

의미:

- 논문 구현 정확도뿐 아니라 retrieval / long-context / serving 관점에서 폭넓게 검증 가능

---

## 7. 현재 저장소에 남아 있는 주요 산출물

### 코드

- `src/turboquant/kv_cache.py`
- `src/turboquant/mlx_quantizer.py`
- `src/turboquant/mlx_attention.py`
- `src/turboquant/metal_kernels.py`
- `src/turboquant/openai_compatible_server.py`

### 실험 스크립트

- `scripts/eval_mlx_turboquant_kv_memory.py`
- `scripts/eval_mlx_vlm_turboquant_kv.py`
- `scripts/eval_mlx_gemma4_turboquant_kv.py`
- `scripts/eval_qwen_wikitext2_ppl.py`
- `scripts/run_qwen_mode_benchmark.py`
- `scripts/run_qwen_vlm_turboquant_sweep.py`
- `scripts/run_gemma4_long_context_throughput.py`
- `scripts/compare_turboquant_plus.py`
- `scripts/serve_openai_gemma4.py`

### 보고서/메모

- `reports/benchmarks/qwen_mode_sweep.csv`
- `reports/benchmarks/qwen_vs_turboquant_plus.csv`
- `reports/benchmarks/qwen_wikitext2_ppl.csv`
- `reports/notes/math_mapping.md`
- `IMPLEMENTATION_PLAN.md`

### 테스트

- `tests/test_kv_cache.py`
- `tests/test_direct_kv_cache.py`
- `tests/test_mlx_attention.py`
- `tests/test_mlx_quantizer.py`
- `tests/test_openai_compatible_server.py`

---

## 8. 현재 워킹트리 상태 요약

`git status --short` 기준으로 다음이 관찰된다.

### 추적 중 파일 수정

- `README.md`
- `scripts/eval_mlx_vlm_turboquant_kv.py`
- `src/turboquant/__init__.py`
- `src/turboquant/kv_cache.py`
- `src/turboquant/metal_kernels.py`
- `src/turboquant/mlx_attention.py`
- `src/turboquant/mlx_quantizer.py`
- `tests/test_direct_kv_cache.py`
- `tests/test_kv_cache.py`
- `tests/test_mlx_attention.py`

### 미추적 파일/디렉터리

- `.continue/`
- `IMPLEMENTATION_PLAN.md`
- `reports/`
- `scripts/compare_turboquant_plus.py`
- `scripts/eval_mlx_gemma4_turboquant_kv.py`
- `scripts/eval_qwen_wikitext2_ppl.py`
- `scripts/run_gemma4_long_context_throughput.py`
- `scripts/run_qwen_mode_benchmark.py`
- `scripts/run_qwen_vlm_turboquant_sweep.py`
- `scripts/serve_openai_gemma4.py`
- `src/turboquant/openai_compatible_server.py`
- `tests/test_mlx_quantizer.py`
- `tests/test_openai_compatible_server.py`

즉, 저장소는 커밋된 기반 위에 상당한 양의 실험/서빙 확장 작업이 얹혀 있는 상태다.

---

## 9. 현재 남아 있는 핵심 이슈

### 9.1 Gemma 4 direct 경로 미완성

문제:

- `direct` 경로가 Gemma 4에서 long/chunked prefill과 mask 정합성 문제를 일으킴

영향:

- 현재 Gemma 4 장문맥 실험은 사실상 `shadow` 경로가 주력

### 9.2 `shadow + dense_shadow`의 메모리 상쇄

문제:

- cache storage는 줄어들어도 dense shadow와 index shadow 때문에 실제 peak memory는 높아질 수 있음

영향:

- “압축했는데 왜 메모리 peak가 높지?” 같은 현상이 발생

### 9.3 README와 최신 상태의 차이

문제:

- README는 존재하지만, 현재 작업트리에 있는 최신 Gemma 4 / 서버 / 벤치마크 확장 상태를 완전히 반영하고 있지는 않음

주의:

- 이 문서는 README보다 최신 작업 상태를 더 정확히 반영한다.

### 9.4 OpenCode 연동 흔적은 현재 저장소에 남아 있지 않음

설명:

- 세션 중 OpenCode 설정 작업은 시도되었지만, 현재 저장소 안에는 해당 설정 파일이 남아 있지 않다.
- 따라서 OpenCode 관련 내용은 “세션 상 시도된 작업”과 “현재 저장소에 남아 있는 산출물”을 구분해서 봐야 한다.

---

## 10. 다음 작업 우선순위 제안

현재 상태를 기준으로 보면 다음 순서가 가장 자연스럽다.

1. Gemma 4 `direct` 경로의 mask/cache bug 수정
2. `shadow` 경로에서 dense shadow / index shadow / packed storage를 분리 계측
3. 장문맥 benchmark 결과를 자동 리포트화
4. Qwen / Gemma 4 / `turboquant_plus` 비교 결과를 공통 포맷으로 정리
5. 필요 시 README 최신화

---

## 11. 결론

현재 저장소는 이미 단순한 논문 재현 코드를 넘어섰다.  
지금의 `Turboquant`는 다음 세 가지 성격을 동시에 가진다.

- **논문 reference implementation**
- **MLX 기반 KV-cache 압축 연구 플랫폼**
- **로컬 Gemma 4/OpenAI-compatible 서빙 실험 환경**

즉, “TurboQuant 수식 구현”에서 출발해서, 지금은 **실모델 long-context 실험과 편집기 연동까지 다루는 로컬 연구용 스택**으로 진화한 상태라고 보는 것이 가장 정확하다.
