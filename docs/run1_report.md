# Run1 — Training Results

## Overview

First training run for ColGemma3 Korean VDR. Establishes the baseline performance on KoViDoRe benchmarks.

- **Model:** [whybe-choi/colgemma3-ko-vdr-v0.3](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-v0.3)
- **Base Model:** google/gemma-3-4b-it (4.37B params)
- **Training Date:** 2026-02-21
- **WandB Run:** [colgemma3-ko-20260221_153705](https://wandb.ai/whybe-choi/ColQwen3-ko/runs/s2msho5s)

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
| Epochs | 1 |
| Per-Device Batch Size | 112 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 896 (112 × 4 GPUs × 2) |
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
| whybe-choi/ko-vdr-train-public-v0.1 | train |
| whybe-choi/ko-vdr-train-private-v0.1 | train |

### Hardware

- **GPU:** 4× NVIDIA RTX PRO 6000 Blackwell (Max-Q Workstation / Server Edition, ~96GB each)
- **CPU:** 32 cores (64 logical)
- **RAM:** ~504GB
- **CUDA:** 13.0
- **Transformers:** 4.57.5

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | jina-embeddings-v4 | TomoroAI |
|------|------|---------------------|----------|
| FinOCR | 0.2027 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | jina-embeddings-v4 | TomoroAI |
|------|------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.5500** | **0.4380** |

## Analysis

### Strengths

- **VQA Retrieval** (0.4804) was the strongest V1 task, suggesting the model handles visual QA-style queries relatively well.
- **Cybersecurity Retrieval** (0.3791) was the strongest V2 task, showing reasonable performance on technical document retrieval.
- **Slide Retrieval** (0.3537) showed moderate performance, likely benefiting from structured visual layouts.

### Weaknesses

- **Economic Retrieval** (0.0565) and **HR Retrieval** (0.0754) were critically low, indicating the model struggles with domain-specific Korean documents.
- **MIR Retrieval** (0.1587) and **FinOCR Retrieval** (0.2027) suggest difficulty with OCR-heavy financial documents.
- Overall performance is far below the jina-embeddings-v4 target (V1: -0.5718, V2: -0.3650).

## Next Steps

- Increase training epochs from 1 to 3 and observe the impact.
- Expand LoRA target modules to include more projection layers while excluding vision encoder:
  `(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj|out_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z).*$|.*(custom_text_proj).*$)`
- Based on the results, add newly synthesized data for further training.
