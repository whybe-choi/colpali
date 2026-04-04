# Run5 — Training Results

## Overview

Fifth training run for ColGemma3 Korean VDR. Run4에서 private 데이터셋 제거로 인한 VQA 성능 급락을 확인한 뒤, private v0.1을 다시 포함하여 성능 회복 여부를 검증하는 런. GradCache 환경(RTX 3090 x2)은 Run4와 동일하게 유지.

- **Model:** [whybe-choi/colgemma3-ko-vdr-v0.7](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-v0.7)
- **Base Model:** google/gemma-3-4b-it (4.37B params)
- **Training Date:** 2026-03-30
- **WandB Run:** [colgemma3-ko-20260330_001451](https://wandb.ai/whybe-choi/Ko-VDR/runs/afiqk4sa)

## Changes from Run4

| Parameter | Run4 | Run5 |
|-----------|------|------|
| Training Dataset | public v1.0 only | **public v1.0 + private v0.1** |

그 외 모든 hyperparameter 및 학습 환경은 Run4와 동일.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-3-4b-it |
| Architecture | Gemma3ForConditionalGeneration |
| Total Parameters | 4.37B |
| Precision | BF16 |
| Attention | flash_attention_2 |
| Loss | ColBERT (temperature=0.02) |
| Optimizer | paged_adamw_8bit |
| Learning Rate | 5e-5 |
| LR Scheduler | linear |
| Warmup Steps | 0.05 |
| Max Grad Norm | 1.0 |
| Weight Decay | 0.0 |
| Epochs | 3 |
| Per-Device Batch Size | 64 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 256 (64 x 2 GPUs x 2) |
| Max Length | 256 |
| Max Visual Tokens | 768 |
| GradCache Mini-Batch Size | 8 |
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
| whybe-choi/ko-vdr-train-public-v1.0 | train |
| whybe-choi/ko-vdr-train-private-v0.1 | train |

### Hardware

- **GPU:** 2x NVIDIA GeForce RTX 3090 (24GB each)
- **CPU:** 16 cores (32 logical)
- **RAM:** ~128GB
- **CUDA:** 12.2
- **Transformers:** 5.3.0
- **PEFT:** 0.18.1

## Training Summary

| Metric | Value |
|--------|-------|
| Train Loss (final) | 0.4624 |
| Train Runtime | 387,262s (~107.6 hours) |
| Samples/second | 2.29 |

Run4(0.6814) 대비 train loss가 0.4624로 크게 낮아졌으며, 학습 시간도 데이터셋 증가에 비례하여 약 64.8h → 107.6h로 증가.

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Δ (Run4→5) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | 0.3411 | 0.5391 | +0.1980 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | 0.2818 | 0.4931 | +0.2113 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | 0.3674 | 0.6005 | +0.2331 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | 0.5077 | 0.7365 | +0.2288 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | 0.4698 | 0.8081 | +0.3383 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **0.3936** | **0.6355** | **+0.2419** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Δ (Run4→5) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | 0.5284 | 0.6003 | +0.0719 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | 0.0630 | 0.1502 | +0.0872 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | 0.3600 | 0.4963 | +0.1363 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | 0.1257 | 0.2706 | +0.1449 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **0.2693** | **0.3793** | **+0.1100** | **0.5500** | **0.4380** |

## Analysis

### VQA 성능 대폭 회복 — Private 데이터 기여 확인

- VQA: 0.4698 → 0.8081 (+0.3383), Run2(0.7416)를 크게 넘어 전체 runs 중 최고 성적.
- Private 데이터셋 재추가만으로 VQA가 이전 최고치를 상회한 것은, private v0.1이 VQA 태스크에 필수적인 학습 signal을 제공하고 있음을 명확히 확인.

### 전체 V1 최고 성능 달성

- V1 average: 0.3936 → 0.6355 (+0.2419), 기존 최고였던 Run2(0.5023)를 크게 상회.
- 모든 V1 태스크에서 Run2 대비 일관되게 향상.

### V2 전반적 향상

- V2 average: 0.2693 → 0.3793 (+0.1100), 전체 runs 중 최고.
- Economic(0.0630→0.1502), HR(0.1257→0.2706) 등 기존에 저성능이었던 태스크에서도 의미 있는 향상.

### Key Takeaways

1. **Private 데이터셋이 핵심:** 데이터셋 구성이 성능에 미치는 영향이 다른 모든 변수를 압도. VQA 성능 0.8081로 전체 runs 최고.
2. **V1/V2 모두 전체 runs 최고 성능:** 현재까지 최선의 설정은 public+private 데이터 + vision encoder 포함 LoRA + GradCache 조합.
3. **다음 과제는 데이터셋 고도화:** 현재 설정의 한계를 넘기 위해서는 데이터 합성 및 hard negative sampling이 핵심 레버.

## Next Steps

- 벤치마크 결과 확인 후 VQA 성능 회복 폭에 따라 데이터셋 전략 확정.
- [ko-vdr-train-public-v2.0](https://huggingface.co/datasets/NomaDamas/ko-vdr-train-public-v2.0)을 활용하여 확장된 public 데이터셋이 성능 향상에 미치는 효과 검증.
- 데이터셋 합성 작업 마무리 후 hard negative sampling 도입.
- Qwen 계열 모델(Qwen2.5-VL 등)을 백본으로 한 학습 실험 진행.
