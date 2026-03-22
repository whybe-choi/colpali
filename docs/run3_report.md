# Run3 — Training Results

## Overview

Third training run for ColGemma3 Korean VDR. LoRA target modules에서 vision encoder 관련 부분을 제외하고, per-device batch size를 112에서 128로 증가.

- **Model:** [whybe-choi/colgemma3-ko-vdr-v0.5](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-v0.5)
- **Base Model:** google/gemma-3-4b-it (4.36B params)
- **Training Date:** 2026-03-21
- **WandB Run:** [colgemma3-ko-20260321_081735](https://wandb.ai/whybe-choi/ColQwen3-ko/runs/5ufu5rhz)

## Changes from Run2

| Parameter | Run2 | Run3 |
|-----------|------|------|
| Per-Device Batch Size | 112 | **128** |
| Effective Batch Size | 448 | **512** (128 x 2 GPUs x 2) |
| LoRA Target Modules | language_model + custom_text_proj | **language_model + custom_text_proj (vision encoder 제외 명시)** |

LoRA target modules regex가 `(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)`로 변경되어 vision encoder의 projection layer들이 학습 대상에서 명시적으로 제외됨. 그 외 hyperparameter는 동일 (epochs=3, lr=5e-5, LoRA r=32 등). Training data도 동일.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-3-4b-it |
| Architecture | Gemma3ForConditionalGeneration |
| Total Parameters | 4.36B |
| Precision | BF16 |
| Attention | flash_attention_2 |
| Loss | ColBERT (temperature=0.02) |
| Optimizer | paged_adamw_8bit |
| Learning Rate | 5e-5 |
| LR Scheduler | linear |
| Warmup Ratio | 0.05 |
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
| Target Modules | `(.*(language_model).*(down_proj\|gate_proj\|up_proj\|k_proj\|q_proj\|v_proj\|o_proj).*$\|.*(custom_text_proj).*$)` |
| Task Type | FEATURE_EXTRACTION |

### Training Data

| Dataset | Split |
|---------|-------|
| whybe-choi/ko-vdr-train-public-v1.0 | train |
| whybe-choi/ko-vdr-train-private-v0.1 | train |

### Hardware

- **GPU:** 2x NVIDIA RTX PRO 6000 Blackwell (Max-Q Workstation / Server Edition, ~96GB each)
- **CPU:** 32 cores (64 logical)
- **RAM:** ~504GB
- **CUDA:** 13.0
- **Transformers:** 5.3.0
- **PEFT:** 0.18.1

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Run3 | Δ (Run2→3) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | -0.0686 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | -0.0737 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | -0.0906 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | -0.0904 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | -0.0638 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **-0.0774** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Δ (Run2→3) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | -0.0566 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | -0.0223 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | -0.0382 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | -0.0409 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **-0.0395** | **0.5500** | **0.4380** |

## Analysis

### Regression from Run2

- **V1 average -15.4%** relative regression (0.5023 → 0.4249). 모든 V1 태스크에서 일관되게 하락.
- **V2 average -14.2%** relative regression (0.2785 → 0.2390). 역시 전 태스크에서 하락.
- Run2 대비 변경 사항: (1) vision encoder LoRA 제외 명시, (2) per-device batch size 112→128.

### Vision Encoder LoRA 제외의 영향

- Run2에서는 LoRA target modules regex가 vision encoder의 projection layer도 매칭할 수 있었으나, Run3에서는 `language_model`에 속한 layer만 명시적으로 매칭.
- Vision encoder를 학습에서 제외하면서, 이미지 특징 추출 능력이 base model 수준에 머물러 retrieval 성능이 전반적으로 하락한 것으로 추정.
- 특히 시각적 레이아웃에 의존하는 **Office** (-0.0906)와 **Slide** (-0.0904)에서 큰 하락을 보인 점이 이를 뒷받침.

### Batch Size 변경의 영향

- Effective batch size가 448→512로 소폭 증가했으나, 이 수준의 차이가 성능에 큰 영향을 미쳤을 가능성은 낮음.
- 성능 하락의 주 원인은 vision encoder LoRA 제외로 판단.

### Key Takeaways

1. **Vision encoder 학습이 중요:** language model만 fine-tuning하는 것보다 vision encoder도 함께 학습시키는 것이 visual document retrieval에서 유의미한 차이를 만듦.
2. **일관된 하락:** 특정 태스크만 하락한 것이 아니라 전 태스크에서 균일하게 하락하여, vision encoder 학습 제외가 전반적인 시각 이해 능력에 영향을 미친 것으로 보임.
3. **Run2(0.5023)가 현재까지 V1 최고 성능:** vision encoder를 포함한 LoRA 학습이 현재 설정에서 더 효과적.

## Next Steps

- Run2 설정(vision encoder 포함) + `custom_text_proj`에 대해서도 LoRA 학습을 적용하는 실험 진행. Batch size는 메모리 상황에 따라 유동적으로 조정.
- 백본 모델을 Qwen/Qwen3-VL-2B-Instruct로 변경하는 실험 진행.
- 학습 데이터셋 확대: ko-vdr-public 데이터 추가 확보 및 각 벤치마크 태스크의 이미지 특성(OCR, 슬라이드, 표 등)에 맞는 공개 영어 VDR 데이터를 학습에 포함시키는 방안 검토.
- Hard negative mining 도입으로 retrieval 품질 개선.
