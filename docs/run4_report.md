# Run4 — Training Results

## Overview

Fourth training run for ColGemma3 Korean VDR. GPU VRAM 24GB 환경(RTX 3090 x2)에서 GradCache를 도입하여 메모리 제약 하에서도 학습이 가능한지 검증하는 런. Per-device batch size를 128에서 64로 축소하는 대신 GradCache(mini-batch-size=8)로 보완. LoRA target modules는 Run3에서 language_model만 명시적으로 학습시키던 방식에서 벗어나 vision encoder를 포함한 전체 projection layer를 학습 대상으로 복귀 (Run2 방식). 데이터셋은 public v1.0만 사용.

- **Model:** [whybe-choi/colgemma3-ko-vdr-v0.6](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-v0.6)
- **Base Model:** google/gemma-3-4b-it (4.37B params)
- **Training Date:** 2026-03-26
- **WandB Run:** [colgemma3-ko-20260326_145404](https://wandb.ai/whybe-choi/Ko-VDR/runs/gvcz83tc/logs?nw=nwuserwhybechoi)

## Changes from Run3

| Parameter | Run3 | Run4 |
|-----------|------|------|
| Hardware | 2x RTX PRO 6000 Blackwell (~96GB each) | **2x NVIDIA RTX 3090 (24GB each)** |
| Per-Device Batch Size | 128 | **64** |
| Effective Batch Size | 512 (128 x 2 GPUs x 2) | **256** (64 x 2 GPUs x 2) |
| LoRA Target Modules | language_model + custom_text_proj (vision encoder 제외) | **all proj layers (vision encoder 포함)** |
| Training Dataset | public v1.0 + private v0.1 | **public v1.0 only** |
| GradCache Mini-Batch | - | **8** |

LoRA target modules가 `custom_text_proj, v_proj, down_proj, o_proj, q_proj, up_proj, k_proj, gate_proj`로 지정되어 vision encoder의 projection layer도 학습 대상에 포함됨 (Run2와 유사). Run3에서 vision encoder를 제외했을 때 성능이 하락한 것을 반영하여 원래대로 복귀.

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
| Train Loss (final) | 0.6814 |
| Train Runtime | 233,409s (~64.8 hours) |
| Samples/second | 2.28 |
| Total Steps | ~2,079 |

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Run3 | Run4 | Δ (Run3→4) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | 0.3411 | +0.0748 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | 0.2818 | -0.0265 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | 0.3674 | +0.0247 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | 0.5077 | -0.0217 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | 0.4698 | -0.2080 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **0.3936** | **-0.0313** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Run4 | Δ (Run3→4) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | 0.5284 | +0.1509 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | 0.0630 | -0.0382 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | 0.3600 | +0.0405 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | 0.1257 | -0.0320 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **0.2693** | **+0.0303** | **0.5500** | **0.4380** |

## Analysis

### V1 vs V2 성능 경향 차이

- **V1 average:** Run3(0.4249) → Run4(0.3936), -7.4% 하락. Run2 대비로는 -21.6%.
- **V2 average:** Run3(0.2390) → Run4(0.2693), +12.7% 상승. Run2 수준(0.2785)에 근접.
- V1과 V2의 성능 방향이 엇갈리는 점이 주목됨.

### VQA 성능 급락

- VQA 태스크에서 0.6778 → 0.4698로 -0.2080 급락, Run1 수준으로 회귀.
- 데이터셋에서 private v0.1을 제거한 것이 VQA 관련 학습 signal 감소로 이어졌을 가능성.
- private 데이터셋은 VQA 태스크 유형의 이미지-쿼리 쌍을 다수 포함하고 있어, 해당 데이터 제거가 직접적인 원인.

### Cybersecurity 성능 대폭 향상

- V2 Cybersecurity: 0.3775 → 0.5284 (+0.1509), 전체 runs 중 최고 성적.
- 사이버보안 문서 특유의 레이아웃 때문이라기보다, V2 Cybersecurity 벤치마크에서 사용된 문서 레이아웃과 유사한 레이아웃의 학습 데이터가 public v1.0에 포함되어 in-batch negative 과정에서 효과적으로 활용된 것으로 추정.

### GradCache 도입 검증

- 24GB VRAM 환경에서 mini-batch-size=8로 GradCache를 적용하여 학습이 정상적으로 완료됨 — 메모리 제약 하에서의 학습 가능성 확인.
- Train loss가 초기 ~5.0에서 최종 0.6814까지 안정적으로 수렴, GradCache 도입으로 인한 학습 불안정 현상은 관찰되지 않음.
- 다만 GradCache를 통한 gradient accumulation이 full-batch gradient와 동일하지 않아 contrastive learning의 negative sample 품질에 영향을 미쳤을 가능성은 배제할 수 없음.

### Effective Batch Size 축소의 영향

- 512 → 256으로 batch size가 절반으로 줄어 contrastive learning에서 배치 내 negative sample 수가 감소.
- 이는 특히 MIR, Slide, VQA처럼 세밀한 시각적 구분이 필요한 태스크에서 부정적 영향을 미쳤을 수 있음.

### 데이터셋 축소의 영향

- Run3 대비 private v0.1 데이터셋을 제외하면서 학습 데이터 다양성 감소.
- VQA 성능 급락과 결합 시, private 데이터의 기여도가 상당했을 가능성이 높음.

### Key Takeaways

1. **GradCache 동작 확인:** 24GB VRAM 환경에서도 GradCache를 통해 안정적인 학습이 가능함을 검증. 향후 동일 환경에서의 학습 기반 확보.
2. **Private 데이터셋의 중요성 재확인:** VQA 성능 급락이 private v0.1 제거에 기인하며, 데이터셋 구성이 성능에 미치는 영향이 가장 큰 변수.
3. **Vision encoder 학습 복귀는 긍정적:** V2 전체 및 일부 V1 태스크(FinOCR, Office)에서 Run3 대비 개선.
4. **다음 방향은 데이터셋 고도화:** GradCache 동작을 확인한 만큼, 학습 환경보다 데이터셋 품질과 구성에 집중하는 방향으로 선회.
5. **백본 모델도 중요한 변수:** TomoroAI는 한국어 학습 데이터 없이도 준수한 성능을 보이는데, 이는 백본 모델 자체의 한국어 이해 및 시각적 표현 능력이 성능에 상당한 영향을 미침을 시사. Qwen 계열 모델에 대한 실험이 필요.

## Next Steps

- Private 데이터셋(ko-vdr-train-private-v0.1)을 다시 포함하여 재학습 — VQA 등 성능이 회복되는지 확인하여 데이터셋 기여도 검증.
- 데이터셋 합성 작업 마무리 후 hard negative sampling 도입 — negative sample 품질 향상으로 contrastive learning 효과 극대화.
- Qwen 계열 모델(Qwen2.5-VL 등)을 백본으로 한 학습 실험 진행 — 한국어 및 시각적 표현 능력이 우수한 백본이 retrieval 성능에 미치는 영향 검증.
