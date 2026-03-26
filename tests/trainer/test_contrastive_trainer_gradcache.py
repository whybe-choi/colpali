import pytest
import torch
from transformers import TrainingArguments

from colpali_engine.loss import ColbertPairwiseCELoss, GradCacheColbertPairwiseCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.float().unsqueeze(-1)
        basis = torch.tensor([1.0, -0.5, 0.75, 1.25], dtype=torch.float32).view(1, 1, 4)
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32).view(1, -1, 1) + 1.0
        return (x + pos * 0.2) * basis * self.weight


class _DummyCollator:
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __call__(self, batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def test_gradcache_rejects_symmetric_loss(tmp_path):
    trainer = ContrastiveTrainer(
        model=_ToyModel(),
        train_dataset=[{"dummy": 1}],
        eval_dataset=None,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            report_to=[],
            use_cpu=True,
        ),
        data_collator=_DummyCollator(),
        loss_func=GradCacheColbertPairwiseCELoss(mini_batch_size=1, temperature=1.0, normalize_scores=False),
        is_vision_model=False,
        compute_symetric_loss=True,
    )

    inputs = {
        "query_input_ids": torch.tensor([[1, 3, 0]], dtype=torch.long),
        "query_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        "doc_input_ids": torch.tensor([[3, 1, 0]], dtype=torch.long),
        "doc_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="do not support compute_symetric_loss"):
        trainer.compute_loss(trainer.model, inputs)


class _DummyTrainDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        samples = [
            {
                "query_input_ids": torch.tensor([1, 3, 0], dtype=torch.long),
                "query_attention_mask": torch.tensor([1, 1, 0], dtype=torch.long),
                "doc_input_ids": torch.tensor([3, 1, 0], dtype=torch.long),
                "doc_attention_mask": torch.tensor([1, 1, 0], dtype=torch.long),
            },
            {
                "query_input_ids": torch.tensor([2, 1, 4], dtype=torch.long),
                "query_attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "doc_input_ids": torch.tensor([1, 4, 2], dtype=torch.long),
                "doc_attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            },
        ]
        return samples[idx]


def test_get_train_dataloader_supports_current_transformers_sampler_signature(tmp_path):
    trainer = ContrastiveTrainer(
        model=_ToyModel(),
        train_dataset=_DummyTrainDataset(),
        eval_dataset=None,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            report_to=[],
            use_cpu=True,
            remove_unused_columns=False,
        ),
        data_collator=_DummyCollator(),
        loss_func=ColbertPairwiseCELoss(temperature=1.0, normalize_scores=False),
        is_vision_model=False,
    )

    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    assert "query_input_ids" in batch
    assert batch["query_input_ids"].shape == (2, 3)


def test_standard_compute_loss_uses_prefixes_without_dataloader_side_effects(tmp_path):
    trainer = ContrastiveTrainer(
        model=_ToyModel(),
        train_dataset=_DummyTrainDataset(),
        eval_dataset=None,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            report_to=[],
            use_cpu=True,
            remove_unused_columns=False,
        ),
        data_collator=_DummyCollator(),
        loss_func=ColbertPairwiseCELoss(temperature=1.0, normalize_scores=False),
        is_vision_model=False,
    )

    inputs = {
        "query_input_ids": torch.tensor([[1, 3, 0], [2, 1, 4]], dtype=torch.long),
        "query_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "doc_input_ids": torch.tensor([[3, 1, 0], [1, 4, 2]], dtype=torch.long),
        "doc_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
    }

    loss = trainer.compute_loss(trainer.model, inputs)

    assert loss.ndim == 0
