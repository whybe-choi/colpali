"""
ColQwen3.5 Korean VDR Training Script

Base model: Qwen/Qwen3.5-VL (via colqwen3_5-base)
Loss: ColBERT cross-entropy with in-batch negatives (temperature=0.02)
LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
                     in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, custom_text_proj

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/configs/qwen3_5/train_colqwen3_5_model.py \
    --output-dir ./outputs/colqwen3_5-ko-run1 \
    --run-name colqwen3_5-ko-run1 \
    --per-device-train-batch-size 64 \
    --peft
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.loss.gradcache_late_interaction_losses import GradCacheColbertLoss, GradCacheColbertPairwiseCELoss
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_hf_datasets

import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5


def _patch_embed_forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(
        -1, self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    )
    weight = self.proj.weight.view(self.embed_dim, -1)
    return F.linear(hidden_states.to(target_dtype), weight, self.proj.bias)


modeling_qwen3_5.Qwen3_5VisionPatchEmbed.forward = _patch_embed_forward
assert modeling_qwen3_5.Qwen3_5VisionPatchEmbed.forward is _patch_embed_forward
print("[patch] Qwen3_5VisionPatchEmbed.forward ->", modeling_qwen3_5.Qwen3_5VisionPatchEmbed.forward.__qualname__)


def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model-name-or-path", type=str, default="./models/base_models/colqwen3_5-base")
    p.add_argument("--max-visual-tokens", type=int, default=768)
    p.add_argument("--max-length", type=int, default=256)

    # Dataset
    p.add_argument("--dataset-name", type=str, nargs="+", required=True, help="one or more HF dataset repo IDs")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--query-column-name", type=str, default="query")
    p.add_argument("--pos-target-column-name", type=str, default="pos")

    # Output
    p.add_argument("--output-dir", type=str, required=True, help="where to write model + script copy")
    p.add_argument("--run-name", type=str, default=None)

    # Model loading
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2")

    # Training
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--warmup-steps", type=float, default=0.05)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--report-to", type=str, default="wandb")

    # Loss
    p.add_argument("--tau", type=float, default=0.02, help="temperature for ColBERT loss")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"])
    p.add_argument("--gradcache", action="store_true", help="use GradCache for memory-efficient training")
    p.add_argument("--mini-batch-size", type=int, default=32, help="sub-batch size for GradCache")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"])

    # LoRA
    p.add_argument("--peft", action="store_true")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj|out_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z).*$|.*(custom_text_proj).*$)",
        help="LoRA target modules regex (peft accepts a regex string).",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Loss function ---
    if args.loss == "ce":
        loss_func = (
            GradCacheColbertLoss(mini_batch_size=args.mini_batch_size, temperature=args.tau)
            if args.gradcache
            else ColbertLoss(
                temperature=args.tau, normalize_scores=True, use_smooth_max=False, pos_aware_negative_filtering=False
            )
        )
    elif args.loss == "pairwise":
        loss_func = (
            GradCacheColbertPairwiseCELoss(mini_batch_size=args.mini_batch_size, temperature=args.tau)
            if args.gradcache
            else ColbertPairwiseCELoss(normalize_scores=False)
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    # --- Load model and processor ---
    model = ColQwen3_5.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    processor = ColQwen3_5Processor.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        max_num_visual_tokens=args.max_visual_tokens,
    )

    # --- Prepare training configuration ---
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        model=model,
        # --- Load training dataset ---
        train_dataset=load_hf_datasets(
            dataset_names=args.dataset_name,
            split=args.train_split,
            query_column_name=args.query_column_name,
            pos_target_column_name=args.pos_target_column_name,
        ),
        max_length=args.max_length,
        run_eval=False,
        loss_func=loss_func,
        # --- Training arguments ---
        tr_args=TrainingArguments(
            output_dir=None,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=args.dataloader_num_workers,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            logging_steps=args.logging_steps,
            warmup_steps=args.warmup_steps,
            learning_rate=args.lr,
            bf16=args.bf16,
            optim=args.optim,
            report_to=args.report_to,
            run_name=args.run_name,
        ),
        # --- LoRA configuration ---
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=args.lora_target_modules,
        )
        if args.peft
        else None,
    )

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    # --- Train and save ---
    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
