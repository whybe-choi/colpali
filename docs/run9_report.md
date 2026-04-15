# Run9 — Training Results

## Overview

Ninth training run for ColQwen2 Korean VDR. 이전 런들과 달리 **vidore/colqwen2-v1.0**을 시작점으로 사용한 첫 번째 2-stage 학습 런. colqwen2-v1.0은 ColPali train set(127K 영어 쌍)으로 이미 학습된 모델이며, 이 위에 한국어 public/private 데이터를 추가 학습(fine-tune)함으로써 영어 VDR 사전 학습의 이점을 한국어 검색에 이식하는 접근을 검증하는 런.

- **Model:** [whybe-choi/colqwen2-ko-vdr-v1.0](https://huggingface.co/whybe-choi/colqwen2-ko-vdr-v1.0)
- **Base Model:** [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) (Qwen2-VL-2B-Instruct 기반, 2.25B params)
- **Training Date:** 2026-04-14
- **WandB Run:** [colqwen2-ko-20260414_184445](https://wandb.ai/whybe-choi/Ko-VDR/runs/svza38ly)

## Training Approach: 2-Stage

| Stage | Model | Data |
|-------|-------|------|
| Stage 1 (선행 학습, vidore 제공) | Qwen2-VL-2B-Instruct → colqwen2-v1.0 | ColPali train set (127K 영어 쌍) |
| Stage 2 (Run9, 한국어 적응) | colqwen2-v1.0 → colqwen2-ko-vdr-v1.0 | NomaDamas/ko-vdr-train-public-v2.0 + whybe-choi/ko-vdr-train-private-v0.1 |

## Changes from Run8

| Parameter | Run8 | Run9 |
|-----------|------|------|
| Base Model | Qwen/Qwen3-VL-4B-Instruct (vanilla) | **vidore/colqwen2-v1.0 (ColPali pre-trained)** |
| Architecture | Qwen3VLForConditionalGeneration | **ColQwen2 (qwen2_vl)** |
| Total Parameters | 4.50B | **2.25B** |
| Pre-training | 없음 (1-stage) | **영어 VDR 선행 학습 (2-stage)** |

학습 데이터, LoRA 설정, 하이퍼파라미터, 하드웨어 환경은 Run8과 동일.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | vidore/colqwen2-v1.0 |
| Architecture | ColQwen2 (Qwen2VLForConditionalGeneration) |
| Total Parameters | 2.25B |
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
| Target Modules | `custom_text_proj, v_proj, down_proj, o_proj, q_proj, up_proj, k_proj, gate_proj` |
| Task Type | FEATURE_EXTRACTION |

### Training Data (Stage 2)

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

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Run7 | Run8 | Run9 | Δ (Run8→9) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------|------|------|------------|---------------------|----------|
| FinOCR | 0.2027 | 0.3349 | 0.2663 | 0.3411 | 0.5391 | 0.5107 | 0.6540 | 0.8225 | 0.7761 | -0.0464 | 0.9410 | 0.8180 |
| MIR | 0.1587 | 0.3820 | 0.3083 | 0.2818 | 0.4931 | 0.5140 | 0.5461 | 0.6359 | 0.7141 | +0.0782 | 0.7360 | 0.6090 |
| Office | 0.2706 | 0.4333 | 0.3427 | 0.3674 | 0.6005 | 0.6562 | 0.7400 | 0.8025 | 0.8529 | +0.0504 | 0.8870 | 0.8420 |
| Slide | 0.3537 | 0.6198 | 0.5294 | 0.5077 | 0.7365 | 0.7066 | 0.7753 | 0.7726 | 0.8614 | +0.0888 | 0.8970 | 0.8630 |
| VQA | 0.4804 | 0.7416 | 0.6778 | 0.4698 | 0.8081 | 0.8140 | 0.8422 | 0.8721 | 0.8669 | -0.0052 | 0.8630 | 0.8290 |
| **Average** | **0.2932** | **0.5023** | **0.4249** | **0.3936** | **0.6355** | **0.6403** | **0.7115** | **0.7811** | **0.8143** | **+0.0332** | **0.8650** | **0.7920** |

### KoViDoRe V2 (NDCG@10)

| Task | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Run7 | Run8 | Run9 | Δ (Run8→9) | jina-embeddings-v4 | TomoroAI |
|------|------|------|------|------|------|------|------|------|------|------------|---------------------|----------|
| Cybersecurity | 0.3791 | 0.4341 | 0.3775 | 0.5284 | 0.6003 | 0.6185 | 0.6489 | 0.7329 | 0.7558 | +0.0229 | 0.7760 | 0.7370 |
| Economic | 0.0565 | 0.1235 | 0.1012 | 0.0630 | 0.1502 | 0.1745 | 0.2079 | 0.2441 | 0.1831 | -0.0610 | 0.2450 | 0.1630 |
| Energy | 0.2290 | 0.3577 | 0.3195 | 0.3600 | 0.4963 | 0.5231 | 0.5498 | 0.6134 | 0.6757 | +0.0623 | 0.6770 | 0.5850 |
| HR | 0.0754 | 0.1986 | 0.1577 | 0.1257 | 0.2706 | 0.3084 | 0.3291 | 0.3271 | 0.4964 | +0.1693 | 0.5010 | 0.2650 |
| **Average** | **0.1850** | **0.2785** | **0.2390** | **0.2693** | **0.3793** | **0.4061** | **0.4339** | **0.4793** | **0.5277** | **+0.0484** | **0.5500** | **0.4380** |

## Analysis

### 2-stage 학습의 효과 확인

- **V1 average:** 0.7811 → 0.8143 (+0.0332), 처음으로 TomoroAI(0.7920)를 명확히 앞섬.
- **V2 average:** 0.4793 → 0.5277 (+0.0484), jina-embeddings-v4(0.5500)와의 Gap이 -0.0223으로 대폭 축소.
- 모델 크기가 4.50B → 2.25B로 절반 수준임에도 전반적으로 더 높은 성능을 달성. 영어 VDR 선행 학습의 효과가 모델 크기 차이를 상쇄함.

### 태스크별 분석

- **Slide(+0.0888), MIR(+0.0782), Office(+0.0504):** 문서 레이아웃 이해 중심 태스크에서 colqwen2-v1.0의 visual representation이 한국어로도 잘 전이됨.
- **HR(+0.1693):** V2에서 가장 큰 향상. 선행 학습에서 익힌 표·텍스트 혼합 문서 이해 능력이 HR 도메인에 특히 유효했던 것으로 보임.
- **FinOCR(-0.0464):** 유일하게 V1에서 하락. Run8의 Qwen3-VL이 Qwen2-VL 대비 OCR 특화 능력이 더 강했던 것으로 추정.
- **Economic(-0.0610):** V2에서 유일하게 하락. 경제 도메인 특화 표현에서 한계가 있거나, Run8에서 우연히 높게 나온 케이스일 가능성.

### 외부 모델 대비 위치

- **V1 average(0.8143):** TomoroAI(0.7920) 대비 +0.0223으로 명확히 앞섬. jina와의 Gap은 -0.0507.
- **V2 average(0.5277):** jina(0.5500)와 Gap이 -0.0223으로 매우 근접. Energy(0.6757 ≈ 0.6770)는 사실상 jina와 동률.

### Key Takeaways

1. **2-stage 학습이 핵심 레버:** 영어 VDR로 선행 학습된 모델 위에 한국어 데이터를 얹는 방식이, 한국어 데이터만으로 1-stage 학습하는 것보다 전반적으로 우세.
2. **파라미터 효율:** 2.25B 모델이 4.50B 1-stage 모델(Run8)보다 전반적으로 높은 성능. 데이터 전략이 모델 크기보다 강한 레버임을 확인.
3. **V2 jina Gap -0.0223:** 목표 달성까지 매우 근접. 2-stage 전략을 계속 개선하면 jina를 넘는 것이 현실적.

## Next Steps

- Qwen3-VL-2B 또는 4B를 베이스로, 공개 영어·다국어 데이터셋(ColPali train set, VisRAG synthetic/in-domain, VDR multilingual 등)으로 Stage 1 학습을 직접 수행한 뒤, 한국어 public/private 데이터로 Stage 2 학습하는 end-to-end 2-stage 파이프라인 구성.
- 한국어 데이터에 한해 hard negative sampling 도입 후 동일 2-stage 설정으로 재비교.
