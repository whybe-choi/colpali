# Run7 — Training Results

## Overview

Seventh training run for ColGemma3 Korean VDR. 정정된 Run6 설정을 바탕으로, B200 환경에서 per-device batch size 128 학습과 GradCache 제거가 실제 성능 향상으로 이어지는지 검증하는 런.

- **Model:** [whybe-choi/colgemma3-ko-vdr-v0.9](https://huggingface.co/whybe-choi/colgemma3-ko-vdr-v0.9)
- **Base Model:** google/gemma-3-4b-it (4.37B params)
- **Training Date:** 2026-04-10
- **WandB Run:** [colgemma3-ko-20260410_002548](https://wandb.ai/whybe-choi/Ko-VDR/runs/wghnrkhe)

## Changes from Run6

| Parameter | Run6 | Run7 |
|-----------|------|------|
| Hardware | 2x NVIDIA RTX 3090 (24GB each) | **B200 environment** |
| Per-Device Batch Size | 64 | **128** |
| Effective Batch Size | 256 (64 x 2 GPUs x 2) | **256** (128 x 1 GPU x 2) |
| GradCache | ✓ (mini-batch=8) | **✗ (제거)** |
| Attention | flash_attention_2 | **sdpa** |

학습 데이터 구성은 Run6와 동일하게 `NomaDamas/ko-vdr-train-public-v2.0 + whybe-choi/ko-vdr-train-private-v0.1`을 사용.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-3-4b-it |
| Architecture | Gemma3ForConditionalGeneration |
| Total Parameters | 4.37B |
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
| Effective Batch Size | 256 (128 x 1 GPU x 2) |
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

- **GPU:** NVIDIA B200 environment
- **Visible GPUs in metadata:** 8x NVIDIA B200 (192GB each)
- **CPU:** 72 cores
- **RAM:** ~2.43TB
- **CUDA:** 13.1
- **Transformers:** 5.5.2
- **PEFT:** 0.18.1

## Benchmark Results

### KoViDoRe V1 (NDCG@5)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Run7 | Δ (Run6→7) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | 0.3411 | 0.5391 | 0.5107 | 0.6540 | +0.1433 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | 0.2818 | 0.4931 | 0.5140 | 0.5461 | +0.0321 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | 0.3674 | 0.6005 | 0.6562 | 0.7400 | +0.0838 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | 0.5077 | 0.7365 | 0.7066 | 0.7753 | +0.0687 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | 0.4698 | 0.8081 | 0.8140 | 0.8422 | +0.0282 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **0.3936** | **0.6355** | **0.6403** | **0.7115** | **+0.0712** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Run7 | Δ (Run6→7) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | 0.5284 | 0.6003 | 0.6185 | 0.6489 | +0.0304 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | 0.0630 | 0.1502 | 0.1745 | 0.2079 | +0.0334 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | 0.3600 | 0.4963 | 0.5231 | 0.5498 | +0.0267 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | 0.1257 | 0.2706 | 0.3084 | 0.3291 | +0.0207 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **0.2693** | **0.3793** | **0.4061** | **0.4339** | **+0.0278** | **0.5500** | **0.4380** |

## Analysis

### Run6 대비 전면적 개선

- **V1 average:** 0.6403 → 0.7115 (+0.0712), 큰 폭으로 상승.
- **V2 average:** 0.4061 → 0.4339 (+0.0278), 전 태스크에서 일관된 향상.
- 정정된 Run6를 기준으로 보면, Run7은 V1/V2 모두에서 뚜렷한 improvement를 보인 런.

### B200 환경 효과 확인

- Run6와 데이터 구성은 동일하지만, B200 환경에서 batch size를 128까지 올리고 GradCache를 제거하면서 성능이 전반적으로 상승.
- 특히 FinOCR(+0.1433), Office(+0.0838), Slide(+0.0687)처럼 문서 레이아웃 의존도가 큰 태스크에서 개선 폭이 큼.
- 이는 더 큰 메모리 여유와 단순화된 학습 경로가 retrieval quality에 긍정적으로 작용했을 가능성을 시사.

### V2도 안정적으로 개선

- Cybersecurity, Economic, Energy, HR가 모두 상승했고, Economic(0.2079)과 HR(0.3291)도 계속 개선.
- 다만 jina-embeddings-v4 대비 격차는 여전히 존재하므로, 도메인 특화 데이터와 학습 전략 개선 여지는 남아 있음.

### Key Takeaways

1. **Run7은 정정된 Run6 대비 명확한 상위 런:** V1 +0.0712, V2 +0.0278로 headline metric이 모두 개선.
2. **B200 환경이 실제로 유의미한 이득을 제공:** 더 큰 batch size와 GradCache 제거가 성능과 학습 단순화 모두에 도움이 됨.
3. **데이터 구성보다 시스템 여건도 중요한 레버:** 동일한 dataset mix에서도 학습 환경 차이만으로 큰 폭의 성능 향상이 가능함을 확인.

## Next Steps

- 영어 및 다국어 공개 문서 검색 데이터셋으로 먼저 학습한 뒤, 한국어 public/private/synthetic 데이터로 이어서 적응시키는 2-stage 학습 실험 진행.
- B200 환경에서 public v2.0 only, public+private, multilingual→Korean 2-stage를 같은 recipe로 비교하는 ablation 진행.
- hard negative sampling 도입 후 동일 설정으로 재비교.
