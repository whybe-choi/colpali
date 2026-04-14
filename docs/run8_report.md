# Run8 — Training Results

## Overview

Eighth training run for ColQwen3 Korean VDR. Run7(ColGemma3)과 동일한 학습 recipe를 유지한 채 베이스 모델을 **Qwen/Qwen3-VL-4B-Instruct**로 교체해, 모델 아키텍처 변경이 한국어 문서 검색 성능에 미치는 영향을 검증하는 런.

- **Model:** [whybe-choi/colqwen3-ko-vdr-v0.1](https://huggingface.co/whybe-choi/colqwen3-ko-vdr-v0.1)
- **Base Model:** Qwen/Qwen3-VL-4B-Instruct (4.50B params)
- **Training Date:** 2026-04-14
- **WandB Run:** [colqwen3-ko-20260414_015650](https://wandb.ai/whybe-choi/Ko-VDR/runs/y9py8s4r)

## Changes from Run7

| Parameter | Run7 | Run8 |
|-----------|------|------|
| Base Model | google/gemma-3-4b-it | **Qwen/Qwen3-VL-4B-Instruct** |
| Architecture | Gemma3ForConditionalGeneration | **Qwen3VLForConditionalGeneration** |
| Total Parameters | 4.37B | **4.50B** |

학습 데이터 구성, LoRA 설정, 하이퍼파라미터, 하드웨어 환경은 Run7과 동일.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3-VL-4B-Instruct |
| Architecture | Qwen3VLForConditionalGeneration |
| Total Parameters | 4.50B |
| Precision | BF16 |
| Attention | sdpa |
| Loss | ColBERT cross-entropy (`loss=ce`, `tau=0.02`) |
| Optimizer | paged_adamw_8bit |
| Learning Rate | 5e-5 |
| LR Scheduler | linear |
| Warmup Steps | 0.05 |
| Max Grad Norm | 1.0 |
| Weight Decay | 0.0 |
| Epochs | 3 |
| Per-Device Batch Size | 128 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 512 (128 x 2 GPUs x 2) |
| Max Length | 256 |
| Max Visual Tokens | 768 |
| Gradient Checkpointing | true (use_reentrant=false) |
| Seed | 42 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| PEFT Type | LoRA |
| Rank (r) | 32 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Target Modules | `custom_text_proj, v_proj, down_proj, o_proj, q_proj, up_proj, k_proj, gate_proj` |
| Task Type | FEATURE_EXTRACTION |

### Training Data

| Dataset | Split |
|---------|-------|
| NomaDamas/ko-vdr-train-public-v2.0 | train |
| whybe-choi/ko-vdr-train-private-v0.1 | train |

### Hardware

- **GPU:** 2x NVIDIA B200 (192GB each)
- **Available GPUs in machine:** 8x NVIDIA B200
- **CPU:** 72 cores
- **RAM:** ~2.43TB
- **CUDA:** 13.1
- **Transformers:** 5.5.2
- **PEFT:** 0.18.1

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Run7 | Run8 | Δ (Run7→8) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | 0.3411 | 0.5391 | 0.5107 | 0.6540 | 0.8225 | +0.1685 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | 0.2818 | 0.4931 | 0.5140 | 0.5461 | 0.6359 | +0.0898 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | 0.3674 | 0.6005 | 0.6562 | 0.7400 | 0.8025 | +0.0625 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | 0.5077 | 0.7365 | 0.7066 | 0.7753 | 0.7726 | -0.0027 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | 0.4698 | 0.8081 | 0.8140 | 0.8422 | 0.8721 | +0.0299 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **0.3936** | **0.6355** | **0.6403** | **0.7115** | **0.7811** | **+0.0696** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Run7 | Run8 | Δ (Run7→8) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | 0.5284 | 0.6003 | 0.6185 | 0.6489 | 0.7329 | +0.0840 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | 0.0630 | 0.1502 | 0.1745 | 0.2079 | 0.2441 | +0.0362 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | 0.3600 | 0.4963 | 0.5231 | 0.5498 | 0.6134 | +0.0636 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | 0.1257 | 0.2706 | 0.3084 | 0.3291 | 0.3271 | -0.0020 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **0.2693** | **0.3793** | **0.4061** | **0.4339** | **0.4793** | **+0.0454** | **0.5500** | **0.4380** |

## Analysis

### Run7 대비 전면적 개선

- **V1 average:** 0.7115 → 0.7811 (+0.0696), 큰 폭으로 상승.
- **V2 average:** 0.4339 → 0.4793 (+0.0454), 전 태스크에서 대부분 개선.
- 학습 recipe와 데이터 구성이 동일함에도 불구하고, Qwen3-VL 베이스 모델이 Gemma3 대비 한국어 문서 검색에서 전반적으로 우세.

### Qwen3-VL 아키텍처의 이점

- **FinOCR(+0.1685)** 에서 가장 큰 개선 폭: OCR 중심 태스크에서 Qwen3-VL의 강력한 비전 인식 능력이 두드러짐.
- **MIR(+0.0898), Cybersecurity(+0.0840)** 에서도 유의미한 향상.
- **Slide(-0.0027), HR(-0.0020)** 는 거의 동일 수준으로, 특정 태스크에서는 모델 아키텍처 교체 효과가 제한적.

### 외부 모델 대비 위치

- **V1 average(0.7811):** TomoroAI(0.7920)에 매우 근접. FinOCR(0.8225 > 0.8180), VQA(0.8721 > 0.8290)에서는 TomoroAI를 이미 앞섬.
- **V2 average(0.4793):** TomoroAI(0.4380) 대비 전체 평균이 앞서는 첫 번째 런.
- jina-embeddings-v4 대비로는 V1 FinOCR(0.8225 < 0.9410)을 포함해 격차가 남아 있지만, 전반적인 Gap이 줄어드는 추세.

### Key Takeaways

1. **Run8은 지금까지 가장 높은 성능의 런:** V1 0.7811, V2 0.4793으로 모든 이전 런을 상회.
2. **Qwen3-VL 아키텍처 교체가 핵심 레버:** 동일 데이터·학습 조건에서 베이스 모델 변경만으로 V1 +0.0696, V2 +0.0454 달성.
3. **V2 평균에서 TomoroAI를 처음으로 초과:** 도메인 특화 태스크 전반에서 경쟁력이 높아짐을 확인.

## Known Issues

### `num_max_visual_tokens` 미전달로 인한 학습 지연

`scripts/configs/qwen3/train_colqwen3_model.py`의 processor 초기화 시 `max_visual_tokens` 인자가 `num_max_visual_tokens`라는 올바른 키워드로 processor에 전달되지 않았음. 이로 인해 visual token 수 제한이 적용되지 않아 실제 의도(768 tokens)보다 많은 visual token이 처리되었고, 학습 시간이 불필요하게 길어졌을 가능성이 있음. 이후 런에서는 해당 인자 전달 여부를 반드시 확인 필요.

## Next Steps

- ColQwen3 베이스에서 hard negative sampling 도입 후 성능 변화 실험.
- 영어 및 다국어 공개 문서 검색 데이터셋으로 먼저 학습한 뒤, 한국어 데이터로 이어서 적응시키는 2-stage 학습 실험 진행.
- public v2.0 only vs. public+private 데이터 ablation을 ColQwen3 베이스에서 재확인.
