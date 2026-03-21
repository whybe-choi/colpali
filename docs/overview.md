# Korean VDR — Project Overview

## Objective

Train a Korean Visual Document Retrieval (VDR) model that surpasses [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) on the [KoViDoRe benchmarks](https://github.com/whybe-choi/kovidore-benchmark).

## Performance Targets

| Benchmark | jina-embeddings-v4 (3.8B) | TomoroAI (8B) | Target |
|-----------|--------------------------|---------------|--------|
| KoViDoRe V1 (NDCG@5) | 0.8650 | 0.7920 | >0.8650 |
| KoViDoRe V2 (NDCG@10) | 0.5500 | 0.4380 | >0.5500 |

### KoViDoRe V1 Per-Task Breakdown (NDCG@5)

| Task | jina-embeddings-v4 (3.8B) | TomoroAI (8B) |
|------|--------------------------|---------------|
| FinOCR | 0.9410 | 0.8180 |
| MIR | 0.7360 | 0.6090 |
| Office | 0.8870 | 0.8420 |
| Slide | 0.8970 | 0.8630 |
| VQA | 0.8630 | 0.8290 |
| **Average** | **0.8650** | **0.7920** |

### KoViDoRe V2 Per-Task Breakdown (NDCG@10)

| Task | jina-embeddings-v4 (3.8B) | TomoroAI (8B) |
|------|--------------------------|---------------|
| Cybersecurity | 0.7760 | 0.7370 |
| Economic | 0.2450 | 0.1630 |
| Energy | 0.6770 | 0.5850 |
| HR | 0.5010 | 0.2650 |
| **Average** | **0.5500** | **0.4380** |

## Technical Foundation

- **Base Model:** google/gemma-3-4b-it or Qwen/Qwen3-VL-2B-Instruct
- **Architecture:** VLM + ColBERT late interaction
- **Loss:** ColBERT cross-entropy with in-batch negatives (temperature=0.02)
- **Fine-tuning:** LoRA (rank 32, alpha 32, dropout 0.05)
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision:** BF16 with gradient checkpointing

## Hardware

- 4× NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM each)
- 32 cores (64 logical), ~504GB RAM

## Training Data

| Dataset | Description |
|---------|-------------|
| whybe-choi/ko-vdr-train-public-v0.1 | Public Korean VDR training data |
| whybe-choi/ko-vdr-train-private-v0.1 | Private Korean VDR training data |

## Competitive Gap Analysis

### jina-embeddings-v4 (3.8B)

- **Base Model:** Qwen2.5-VL-3B-Instruct
- **Dual embedding:** Single-vector (2048-dim, Matryoshka truncatable) + Multi-vector (128-dim) late interaction
- **Task-specific LoRA adapters:** 3 adapters (retrieval, text-matching, code) with ~60M params each
- **Training:** 2-stage — (1) joint pair training combining single/multi-vector objectives, (2) task-specific adapter training
- **Loss:** Joint InfoNCE + KL divergence for single/multi-vector alignment
- **Cross-modal alignment:** 0.71 alignment score (vs 0.15 for OpenAI CLIP) via unified encoder
- **Context:** 32K tokens, M-RoPE positional encoding

### TomoroAI/tomoro-colqwen3-embed-8b (8B)

- **Base Model:** Merged from Qwen3-VL-8B-Instruct + Qwen3-Embedding-8B (text embedding model merge)
- **Projection:** 320-dim multi-vector embeddings with MaxSim scoring
- **Training data:** VDR multilingual, ColPali train set, VisRAG synthetic + in-domain data
- **Visual tokens:** Up to 1,280 per page

### Key differences from our approach

| Aspect | Ours (Run1) | jina-embeddings-v4 | TomoroAI |
|--------|-------------|---------------------|----------|
| Base Model | Gemma3 4B | Qwen2.5-VL 3B | Qwen3-VL 8B + Qwen3-Embedding 8B (merged) |
| Embedding | Multi-vector only | Single + Multi-vector | Multi-vector (320-dim) |
| Training stages | 1-stage | 2-stage (joint + adapter) | Unknown |
| Loss | ColBERT CE | InfoNCE + KL divergence | ColBERT (ColPali-style) |
| Text embedding init | None | Unknown | Merged from text embedding model |
| Training data | Korean VDR only | Large-scale multilingual | Multilingual VDR + synthetic |

## Run History

| Run | Model | V1 Avg | V2 Avg | Notes |
|-----|-------|--------|--------|-------|
| [Run1](run1_report.md) | colgemma3-ko-vdr-v0.3 | 0.2932 | 0.1850 | Baseline |
| [Run2](run2_report.md) | colgemma3-ko-vdr-0.4 | 0.5023 | 0.2785 | Epoch 1→3, public data v0.1→v1.0 |

## Next Steps

- Expand training data (synthetic QA generation)
- Implement hard negative mining
- Hyperparameter tuning (LR, LoRA rank, batch size)
- Multi-epoch training experiments
