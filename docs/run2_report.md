# Run2 — Training Results

## Overview

Second training run for ColGemma3 Korean VDR. Epoch count increased from 1 to 3 to observe the impact of longer training on KoViDoRe benchmarks.

- **Model:** [whybe-choi/colgemma3-ko-vdr-0.4](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-0.4)
- **Base Model:** google/gemma-3-4b-it (4.37B params)
- **Training Date:** 2026-03-19
- **WandB Run:** [colgemma3-ko-20260319_135501](https://wandb.ai/whybe-choi/ColQwen3-ko/runs/8eonrwvu1jyykyvzb4u36h4onqhay7rp)

## Changes from Run1

| Parameter | Run1 | Run2 |
|-----------|------|------|
| Epochs | 1 | **3** |
| GPUs | 4 | **2** |
| Effective Batch Size | 896 | **448** |
| Transformers | 4.57.5 | **5.3.0** |
| PEFT | — | **0.18.1** |

Training data는 동일 (ko-vdr-train-public-v0.1 = v1.0, 버전명만 변경). 그 외 hyperparameter도 동일.

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
| Warmup Ratio | 0.05 |
| Max Grad Norm | 1.0 |
| Weight Decay | 0.0 |
| Epochs | 3 |
| Per-Device Batch Size | 112 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 448 (112 × 2 GPUs × 2) |
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
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Task Type | FEATURE_EXTRACTION |

### Training Data

| Dataset | Split |
|---------|-------|
| whybe-choi/ko-vdr-train-public-v1.0 | train |
| whybe-choi/ko-vdr-train-private-v0.1 | train |

### Hardware

- **GPU:** 2× NVIDIA RTX PRO 6000 Blackwell (Max-Q Workstation / Server Edition, ~96GB each)
- **CPU:** 32 cores (64 logical)
- **RAM:** ~504GB
- **CUDA:** 13.0
- **Transformers:** 5.3.0

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Δ | jina-embeddings-v4 | TomoroAI |
|------|------|------|---|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | +0.1322 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | +0.2233 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | +0.1627 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | +0.2661 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | +0.2612 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **+0.2091** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Δ | jina-embeddings-v4 | TomoroAI |
|------|------|------|---|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | +0.0550 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | +0.0670 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | +0.1287 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | +0.1232 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **+0.0935** | **0.5500** | **0.4380** |

## Analysis

### Improvements

- **V1 average +71.3%** relative improvement (0.2932 → 0.5023). Epoch 증가와 데이터셋 업데이트가 전반적으로 큰 효과를 보임.
- **VQA Retrieval** (0.7416)이 jina-embeddings-v4 대비 86% 수준까지 도달, 가장 gap이 작은 태스크.
- **Slide Retrieval** (+0.2661)과 **VQA Retrieval** (+0.2612)에서 가장 큰 절대 향상을 기록.
- **MIR Retrieval** (+0.2233)이 가장 높은 상대적 개선율 (140.7%).

### Remaining Gaps

- **V1 gap:** target 대비 -0.3627 (0.5023 vs 0.8650). 여전히 jina-embeddings-v4의 58% 수준.
- **V2 gap:** target 대비 -0.2715 (0.2785 vs 0.5500). jina-embeddings-v4의 50.6% 수준.
### Training Loss 분석

- Loss가 epoch ~0.8 부근에서 이미 수렴하여 이후 2 epoch 동안 거의 변화 없음 (최종 loss ~0.044).
- 현재 설정에서 epoch을 추가로 늘려도 (e.g. 5 epoch) loss 개선은 미미할 것으로 판단되며, overfitting 리스크만 증가.
- **결론:** epoch 증가를 통한 성능 향상은 한계에 도달. 추가 개선은 백본 변경이나 데이터셋 확대를 통해 추구해야 함.

### Key Takeaways

1. **Epoch 증가 효과 확인:** 1→3 epoch만으로 V1 +0.2091, V2 +0.0935의 유의미한 향상.
2. **GPU 절반으로도 유의미한 향상:** effective batch size가 896→448로 줄었음에도 epoch 증가로 충분히 보상.
3. **V2 개선폭이 V1보다 작음:** V2 태스크들이 더 도메인 특화적이어서 일반적인 학습량 증가만으로는 한계.
4. **Loss 수렴 확인:** epoch ~0.8에서 이미 수렴하여, 추가 epoch 증가보다는 백본 변경 및 데이터셋 확대가 우선.

## Next Steps

- 백본 모델을 Qwen/Qwen3-VL-2B-Instruct로 변경.
- 학습 데이터셋 확대.
- LoRA target modules 확장 실험.
- Hard negative mining 도입으로 retrieval 품질 개선.
