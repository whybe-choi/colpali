# Korean VDR — Project Overview

## Objective

Train a Korean Visual Document Retrieval (VDR) model that surpasses [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) on the [KoViDoRe benchmarks](https://github.com/whybe-choi/kovidore-benchmark).

## Current Status

| Benchmark | Current Best Ours (Run9) | jina-embeddings-v4 (3.8B) | TomoroAI (8B) | Gap to Best |
|-----------|--------------------------|----------------------------|---------------|-------------|
| KoViDoRe V1 (NDCG@5) | 0.8143 | 0.8650 | 0.7920 | -0.0507 vs jina |
| KoViDoRe V2 (NDCG@10) | 0.5277 | 0.5500 | 0.4380 | -0.0223 vs jina |

Run9 기준으로 V1/V2 모두 현재까지 최고 성능을 달성했다. colqwen2-v1.0(영어 VDR 선행 학습 모델) 위에 한국어 데이터를 fine-tune하는 2-stage 접근이 핵심이었다. V1 평균(0.8143)은 TomoroAI(0.7920)를 처음으로 명확히 앞섰고, V2 평균(0.5277)은 jina와의 Gap이 -0.0223까지 축소됐다.

## Performance Targets

| Benchmark | jina-embeddings-v4 (3.8B) | TomoroAI (8B) | Target |
|-----------|--------------------------|---------------|--------|
| KoViDoRe V1 (NDCG@5) | 0.8650 | 0.7920 | >0.8650 |
| KoViDoRe V2 (NDCG@10) | 0.5500 | 0.4380 | >0.5500 |

### KoViDoRe V1 Per-Task Breakdown (NDCG@5)

| Task | Current Best Ours (Run9) | jina-embeddings-v4 (3.8B) | TomoroAI (8B) |
|------|--------------------------|---------------------------|---------------|
| FinOCR | 0.7761 | 0.9410 | 0.8180 |
| MIR | 0.7141 | 0.7360 | 0.6090 |
| Office | 0.8529 | 0.8870 | 0.8420 |
| Slide | 0.8614 | 0.8970 | 0.8630 |
| VQA | 0.8669 | 0.8630 | 0.8290 |
| **Average** | **0.8143** | **0.8650** | **0.7920** |

### KoViDoRe V2 Per-Task Breakdown (NDCG@10)

| Task | Current Best Ours (Run9) | jina-embeddings-v4 (3.8B) | TomoroAI (8B) |
|------|--------------------------|---------------------------|---------------|
| Cybersecurity | 0.7558 | 0.7760 | 0.7370 |
| Economic | 0.1831 | 0.2450 | 0.1630 |
| Energy | 0.6757 | 0.6770 | 0.5850 |
| HR | 0.4964 | 0.5010 | 0.2650 |
| **Average** | **0.5277** | **0.5500** | **0.4380** |

## Technical Foundation

- **Current base model:** vidore/colqwen2-v1.0 (Qwen2-VL-2B-Instruct 기반)
- **Architecture:** VLM + ColBERT late interaction
- **Loss:** ColBERT cross-entropy with in-batch negatives (`tau=0.02`)
- **Fine-tuning:** LoRA (`r=32`, `alpha=32`, `dropout=0.05`)
- **Target modules:** `custom_text_proj, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Precision:** BF16 with gradient checkpointing
- **Optim:** `paged_adamw_8bit`

## Training Environments

- **Run1-3:** High-memory Blackwell environment
- **Run4-6:** 2x NVIDIA RTX 3090 (24GB each) with GradCache-based training
- **Run7:** NVIDIA B200 environment with larger batch size and no GradCache
- **Run8:** 2x NVIDIA B200, ColQwen3 (Qwen3-VL-4B-Instruct) base model
- **Run9:** 2x NVIDIA B200, 첫 2-stage 학습 (colqwen2-v1.0 → 한국어 fine-tune)

## Training Data

| Dataset | Role in experiments |
|---------|---------------------|
| whybe-choi/ko-vdr-train-public-v0.1 | Early public baseline data |
| whybe-choi/ko-vdr-train-public-v1.0 | Main public dataset for Run4-5 |
| NomaDamas/ko-vdr-train-public-v2.0 | Expanded public dataset used in Run6-7; includes v1.0 |
| whybe-choi/ko-vdr-train-private-v0.1 | Private Korean VDR data used in Run1-3 and Run5-7 |

## Competitive Gap Analysis

### jina-embeddings-v4 (3.8B)

- **Base Model:** Qwen2.5-VL-3B-Instruct
- **Dual embedding:** Single-vector (2048-dim, Matryoshka truncatable) + Multi-vector (128-dim) late interaction
- **Task-specific LoRA adapters:** 3 adapters (retrieval, text-matching, code) with ~60M params each
- **Training:** 2-stage joint training + adapter specialization
- **Loss:** Joint InfoNCE + KL divergence for single/multi-vector alignment
- **Cross-modal alignment:** Strong unified text-image representation
- **Context:** 32K tokens, M-RoPE positional encoding

### TomoroAI/tomoro-colqwen3-embed-8b (8B)

- **Base Model:** Qwen3-VL-8B-Instruct + Qwen3-Embedding-8B merge
- **Projection:** 320-dim multi-vector embeddings with MaxSim scoring
- **Training data:** Multilingual VDR, ColPali train set, VisRAG synthetic + in-domain data
- **Visual tokens:** Up to 1,280 per page

### Key differences from our current approach

| Aspect | Ours (Run9) | jina-embeddings-v4 | TomoroAI |
|--------|-------------|---------------------|----------|
| Base Model | Qwen2-VL 2B (2-stage) | Qwen2.5-VL 3B | Qwen3-VL 8B + Qwen3-Embedding 8B |
| Embedding | Multi-vector only | Single + Multi-vector | Multi-vector (320-dim) |
| Training stages | 2-stage | 2-stage | Unknown |
| Loss | ColBERT CE | InfoNCE + KL divergence | ColBERT-style |
| Text embedding init | None | Unknown | Merged from text embedding model |
| Training data | English ColPali → Korean public/private | Large-scale multilingual | Multilingual VDR + synthetic |

## Run History

| Run | Model | V1 Avg | V2 Avg | Notes |
|-----|-------|--------|--------|-------|
| [Run1](run1_report.md) | colgemma3-ko-vdr-v0.3 | 0.2932 | 0.1850 | Baseline |
| [Run2](run2_report.md) | colgemma3-ko-vdr-v0.4 | 0.5023 | 0.2785 | Epoch 1→3 확장으로 큰 폭 개선 |
| [Run3](run3_report.md) | colgemma3-ko-vdr-v0.5 | 0.4249 | 0.2390 | Vision encoder LoRA 제외, 성능 하락 |
| [Run4](run4_report.md) | colgemma3-ko-vdr-v0.6 | 0.3936 | 0.2693 | 2x3090 + GradCache, public-only |
| [Run5](run5_report.md) | colgemma3-ko-vdr-v0.7 | 0.6355 | 0.3793 | Private v0.1 복귀로 성능 대폭 회복 |
| [Run6](run6_report.md) | colgemma3-ko-vdr-v0.8 | 0.6403 | 0.4061 | public v2.0 도입, V2 중심 개선 |
| [Run7](run7_report.md) | colgemma3-ko-vdr-v0.9 | 0.7115 | 0.4339 | B200 환경, batch 128, GradCache 제거 |
| [Run8](run8_report.md) | colqwen3-ko-vdr-v0.1 | 0.7811 | 0.4793 | ColQwen3로 베이스 모델 교체 |
| [Run9](run9_report.md) | colqwen2-ko-vdr-v1.0 | 0.8143 | 0.5277 | 첫 2-stage 학습 (colqwen2-v1.0 → 한국어 fine-tune), 현재 최고 성능 |

## Trend Summary

- **가장 큰 성능 레버는 데이터 구성:** Run4→Run5에서 private 데이터 복귀만으로 큰 폭의 회복이 발생했다.
- **public v2.0은 V2 generalization에 특히 효과적:** Run6에서 V1은 mixed result였지만 V2는 전반적으로 개선됐다.
- **시스템 여건도 중요한 변수:** Run7에서 B200 환경, 큰 batch size, GradCache 제거가 추가 향상으로 이어졌다.
- **베이스 모델도 중요한 레버:** Run8에서 ColQwen3 교체만으로 V1 +0.0696, V2 +0.0454를 달성했다.
- **2-stage 학습이 현재 가장 강한 레버:** Run9에서 영어 VDR 선행 학습 모델 위에 한국어 데이터를 얹는 방식이, 더 큰 모델로 1-stage 학습하는 것보다 V1/V2 모두에서 우세함을 확인했다.
- **jina와의 격차가 빠르게 좁혀지고 있음:** V2 Gap이 -0.0223까지 축소됐고, Energy 태스크는 사실상 동률이다.

## Next Steps

- Qwen3-VL-2B 또는 4B를 베이스로, 공개 영어·다국어 데이터셋(ColPali train set, VisRAG synthetic/in-domain, VDR multilingual 등)으로 Stage 1 학습을 직접 수행한 뒤, 한국어 public/private 데이터로 Stage 2 학습하는 end-to-end 2-stage 파이프라인 구성.
- 한국어 데이터에 한해 hard negative sampling 도입 후 동일 2-stage 설정으로 재비교.
