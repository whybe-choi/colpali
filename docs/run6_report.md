# Run6 — Training Results

## Overview

Sixth training run for ColGemma3 Korean VDR. Run5의 학습 recipe는 유지한 채 public 데이터셋만 `ko-vdr-train-public-v2.0`으로 교체해, 확장된 public 데이터가 성능에 미치는 영향을 검증하는 런.

- **Model:** [whybe-choi/colgemma3-ko-vdr-v0.8](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-v0.8)
- **Base Model:** google/gemma-3-4b-it (4.37B params)
- **Training Date:** 2026-04-05
- **WandB Run:** [colgemma3-ko-20260405_014956](https://wandb.ai/whybe-choi/Ko-VDR/runs/aso5oxyv)

## Changes from Run5

| Parameter | Run5 | Run6 |
|-----------|------|------|
| Public Dataset | whybe-choi/ko-vdr-train-public-v1.0 | **NomaDamas/ko-vdr-train-public-v2.0** |
| Positive Column | `pos` | **`image`** |

그 외 hyperparameter, LoRA 설정, hardware, GradCache 구성은 Run5와 동일.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-3-4b-it |
| Architecture | Gemma3ForConditionalGeneration |
| Total Parameters | 4.37B |
| Precision | BF16 |
| Attention | flash_attention_2 |
| Loss | ColBERT cross-entropy (`loss=ce`, `tau=0.02`) |
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
| GradCache | true |
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
| NomaDamas/ko-vdr-train-public-v2.0 | train |
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
| Train Loss (final) | 0.3218 |
| Train Runtime | 647,134s (~179.8 hours) |
| Samples/second | 1.99 |
| Steps/second | 0.008 |

Run5(0.4624) 대비 train loss는 더 낮아졌지만, 학습 시간은 public v2.0 데이터셋 확장에 따라 약 107.6h → 179.8h로 크게 증가.

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Δ (Run5→6) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | 0.3411 | 0.5391 | 0.5107 | -0.0284 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | 0.2818 | 0.4931 | 0.5140 | +0.0209 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | 0.3674 | 0.6005 | 0.6562 | +0.0557 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | 0.5077 | 0.7365 | 0.7066 | -0.0299 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | 0.4698 | 0.8081 | 0.8140 | +0.0059 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **0.3936** | **0.6355** | **0.6403** | **+0.0048** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Δ (Run5→6) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | 0.5284 | 0.6003 | 0.6185 | +0.0182 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | 0.0630 | 0.1502 | 0.1745 | +0.0243 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | 0.3600 | 0.4963 | 0.5231 | +0.0268 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | 0.1257 | 0.2706 | 0.3084 | +0.0378 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **0.2693** | **0.3793** | **0.4061** | **+0.0268** | **0.5500** | **0.4380** |

## Analysis

### Public v2.0 효과는 V2에서 더 뚜렷

- **V1 average:** 0.6355 → 0.6403 (+0.0048), 소폭 개선.
- **V2 average:** 0.3793 → 0.4061 (+0.0268), 전 태스크에서 일관된 향상.
- public v2.0 교체 효과는 V1 headline improvement보다 V2 generalization 개선에서 더 크게 나타남.

### V1은 mixed result

- Office(+0.0557), MIR(+0.0209), VQA(+0.0059)는 개선.
- 반면 FinOCR(-0.0284), Slide(-0.0299)는 하락.
- 즉, public v2.0이 모든 V1 태스크에 일괄적으로 유리하지는 않았고, 태스크별 trade-off가 존재.

### V2는 전체적으로 안정적 개선

- Cybersecurity, Economic, Energy, HR 모두 Run5 대비 상승.
- 특히 HR(+0.0378), Energy(+0.0268), Economic(+0.0243)에서 개선 폭이 상대적으로 큼.
- TomoroAI 대비 **Economic(0.1745 > 0.1630)**, **HR(0.3084 > 0.2650)**는 앞섰지만, 전체 평균은 아직 뒤처짐.

### Key Takeaways

1. **public v2.0은 유효하지만 효과가 선택적:** Run5 대비 V1은 거의 비슷한 수준, V2는 의미 있게 개선.
2. **데이터 교체만으로는 전면적 개선이 어렵다:** FinOCR, Slide 하락은 데이터 mix와 샘플 분포 재조정 필요성을 시사.
3. **다음 레버는 데이터 mixing과 hard negative:** public v1.0/v2.0 혼합, task-balanced sampling, hard negative 도입이 다음 실험 포인트.

## Next Steps

- 영어 및 다국어 공개 문서 검색 데이터셋으로 먼저 학습한 뒤, 한국어 public/private/synthetic 데이터로 이어서 적응시키는 2-stage 학습 실험 진행.
- 새로 확보한 B200 환경에서 더 큰 per-device batch size와 GradCache 제거 가능성을 포함해 동일 recipe를 재검증.
- hard negative sampling 도입 후 동일 설정으로 재비교.
