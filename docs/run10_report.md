# Run10 — Training Results

## Overview

Tenth training run for ColQwen3 Korean VDR. Run9의 2-stage 전략을 Qwen3-VL-4B-Instruct 기반으로 **완전히 자체 구현한 end-to-end 2-stage 파이프라인** 첫 번째 런. Run9이 vidore가 제공한 colqwen2-v1.0(외부 선행 학습 모델)을 Stage 1로 사용했다면, Run10은 공개 영어·다국어 데이터셋으로 Stage 1을 자체 학습한 뒤 한국어 데이터로 Stage 2를 적용한다.

- **Model:** [whybe-choi/colqwen3-ko-vdr-v1.0](https://huggingface.co/whybe-choi/colqwen3-ko-vdr-v1.0)
- **Base (Stage 1):** [whybe-choi/colqwen3-ko-vdr-base](https://huggingface.co/whybe-choi/colqwen3-ko-vdr-base)
- **Training Date:** 2026-04-16
- **WandB Run (Stage 1):** [colqwen3-ko-20260416_024626](https://wandb.ai/whybe-choi/Ko-VDR/runs/hafn47zv)
- **WandB Run (Stage 2):** [colqwen3-ko-20260416_142533](https://wandb.ai/whybe-choi/Ko-VDR/runs/k2l8zevr)

## Training Approach: 2-Stage (End-to-End Self-Trained)

| Stage | Model | Data | Duration |
|-------|-------|------|----------|
| Stage 1 (다국어 VDR 선행 학습) | Qwen/Qwen3-VL-4B-Instruct → colqwen3-ko-vdr-base | ColPali (127K) + VisRAG Synthetic (239K) + VisRAG In-domain (123K) + VDR Multilingual en/de/es/fr/it | ~8.9h |
| Stage 2 (한국어 적응) | colqwen3-ko-vdr-base → colqwen3-ko-vdr-v1.0 | NomaDamas/ko-vdr-train-public-v2.0 + whybe-choi/ko-vdr-train-private-v0.1 | ~15.6h |

## Changes from Run9

| Parameter | Run9 | Run10 |
|-----------|------|-------|
| Base (Stage 1) | vidore/colqwen2-v1.0 (외부 제공) | **Qwen/Qwen3-VL-4B-Instruct (자체 학습)** |
| Architecture | ColQwen2 (qwen2_vl) | **ColQwen3 (qwen3_vl)** |
| Total Parameters | 2.25B | **4.50B** |
| Stage 1 Data | ColPali train set (외부) | **ColPali + VisRAG + VDR multilingual (자체 구성)** |
| GPU | 2x B200 | **2x B200** |
| Effective Batch Size | 512 (128×2×2) | **512 (128×2×2)** |

## Training Configuration

### Stage 1 (colqwen3-ko-vdr-base)

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3-VL-4B-Instruct |
| Architecture | ColQwen3 (Qwen3VLForConditionalGeneration) |
| Total Parameters | 4.50B |
| Precision | BF16 |
| Attention | sdpa |
| Loss | ColBERT cross-entropy (`loss=ce`, `tau=0.02`) |
| Optimizer | paged_adamw_8bit |
| Learning Rate | 5e-5 |
| LR Scheduler | linear |
| Warmup Steps | 0.05 |
| Epochs | 1 |
| Per-Device Batch Size | 128 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 512 (128 × 2 GPUs × 2) |
| Max Length | 256 |
| Max Visual Tokens | 768 |
| Total Steps | 1,487 |
| Final Loss | 0.0780 |
| Gradient Checkpointing | true (use_reentrant=false) |
| Seed | 42 |

### Stage 2 (colqwen3-ko-vdr-v1.0)

| Parameter | Value |
|-----------|-------|
| Base Model | whybe-choi/colqwen3-ko-vdr-base |
| Precision | BF16 |
| Attention | sdpa |
| Loss | ColBERT cross-entropy (`loss=ce`, `tau=0.02`) |
| Optimizer | paged_adamw_8bit |
| Learning Rate | 5e-5 |
| LR Scheduler | linear |
| Warmup Steps | 0.05 |
| Epochs | 3 |
| Per-Device Batch Size | 128 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 512 (128 × 2 GPUs × 2) |
| Max Length | 256 |
| Max Visual Tokens | 768 |
| Total Steps | 2,514 |
| Final Loss | 0.2823 |
| Gradient Checkpointing | true (use_reentrant=false) |
| Seed | 42 |

### LoRA Configuration (Stage 1 & 2 공통)

| Parameter | Value |
|-----------|-------|
| PEFT Type | LoRA |
| Rank (r) | 32 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Target Modules | `custom_text_proj, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Task Type | FEATURE_EXTRACTION |

### Training Data

**Stage 1 (다국어 VDR 선행 학습):**

| Dataset | Pairs |
|---------|-------|
| vidore/colpali_train_set | ~127K |
| openbmb/VisRAG-Ret-Train-Synthetic-data | ~239K |
| openbmb/VisRAG-Ret-Train-In-domain-data | ~123K |
| llamaindex/vdr-multilingual-train (en/de/es/fr/it) | 5개 언어 |

**Stage 2 (한국어 적응):**

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

| Task | Run8 | Run9 | Run10 | Δ (Run9→10) | jina-embeddings-v4 | TomoroAI |
|------|------|------|-------|-------------|---------------------|----------|
| FinOCR | 0.8225 | 0.7761 | 0.8229 | +0.0468 | 0.9410 | 0.8180 |
| MIR | 0.6359 | 0.7141 | 0.6599 | -0.0542 | 0.7360 | 0.6090 |
| Office | 0.8025 | 0.8529 | 0.7797 | -0.0732 | 0.8870 | 0.8420 |
| Slide | 0.7726 | 0.8614 | 0.8529 | -0.0085 | 0.8970 | 0.8630 |
| VQA | 0.8721 | 0.8669 | 0.8835 | +0.0166 | 0.8630 | 0.8290 |
| **Average** | **0.7811** | **0.8143** | **0.7998** | **-0.0145** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run8 | Run9 | Run10 | Δ (Run9→10) | jina-embeddings-v4 | TomoroAI |
|------|------|------|-------|-------------|---------------------|----------|
| Cybersecurity | 0.7329 | 0.7558 | 0.7618 | +0.0060 | 0.7760 | 0.7370 |
| Economic | 0.2441 | 0.1831 | 0.2286 | +0.0455 | 0.2450 | 0.1630 |
| Energy | 0.6134 | 0.6757 | 0.6577 | -0.0180 | 0.6770 | 0.5850 |
| HR | 0.3271 | 0.4964 | 0.4786 | -0.0178 | 0.5010 | 0.2650 |
| **Average** | **0.4793** | **0.5277** | **0.5317** | **+0.0040** | **0.5500** | **0.4380** |

## Analysis

### Run9 대비 혼재된 결과

- **V1 average:** 0.8143 → 0.7998 (-0.0145) — 소폭 하락, 현재 최고 성능은 Run9.
- **V2 average:** 0.5277 → 0.5317 (+0.0040) — 소폭 개선.

전반적으로 Run9(2.25B, vidore 외부 선행 학습)과 Run10(4.5B, 자체 선행 학습)이 비슷한 수준. 자체 Stage 1 학습의 방향성 자체는 유효하나 아직 vidore의 colqwen2-v1.0이 제공하는 선행 학습 품질을 완전히 복제하지 못한 상태.

### 태스크별 분석

- **FinOCR(+0.0468):** Qwen3-VL의 강점인 OCR 능력이 Stage 1 학습 후에도 유지. Run8 대비도 +0.0004로 사실상 동률.
- **VQA(+0.0166):** VQA 태스크에서 Qwen3-VL 기반이 Qwen2-VL 대비 더 강한 시각 이해 능력을 발휘.
- **Economic(+0.0455):** Run9의 Economic 하락(-0.0610)이 어느 정도 회복. Qwen3-VL의 다국어 이해 능력이 경제 도메인 한국어 문서에 도움이 된 것으로 추정.
- **Office(-0.0732), MIR(-0.0542):** V1 하락의 주요 원인. Run9(colqwen2-v1.0)에서 문서 레이아웃 이해 능력이 더 효과적으로 전이된 영역에서 자체 Stage 1 학습이 미치지 못함.

### 외부 모델 대비 위치

- **V1 average(0.7998):** TomoroAI(0.7920) 대비 +0.0078로 앞서 있으나, Run9(0.8143) 대비 하락.
- **V2 average(0.5317):** jina(0.5500)와 Gap이 -0.0183으로 Run9(-0.0223)보다 소폭 축소.

### Key Takeaways

1. **자체 Stage 1 학습의 방향성은 유효:** End-to-end 2-stage 파이프라인이 1-stage(Run8, 0.7811) 대비 V1/V2 모두 개선.
2. **Stage 1 품질이 관건:** vidore/colqwen2-v1.0 기반 Run9을 V1에서 넘지 못함. Stage 1 학습 데이터 구성, epoch 수, 배치 크기 등 추가 최적화 여지.
3. **Qwen3-VL의 강점 태스크가 명확:** FinOCR, VQA는 Qwen3-VL 기반이 Qwen2-VL 기반보다 일관되게 강함. 레이아웃·구조 이해(Office, MIR)는 여전히 개선 필요.
4. **V2 jina Gap -0.0183:** Run9(-0.0223)보다 좁혀짐. Economic 회복이 기여.

## Next Steps

- 한국어 데이터에 한해 hard negative sampling 도입 후 동일 2-stage 설정으로 재비교.
