from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.nn.functional import all_gather

from colpali_engine.loss import (
    ColbertLoss,
    ColbertPairwiseCELoss,
    ColbertPairwiseNegativeCELoss,
    GradCacheColbertLoss,
    GradCacheColbertPairwiseCELoss,
    GradCacheColbertPairwiseNegativeCELoss,
)


class _ToyTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.float().unsqueeze(-1)
        basis = torch.tensor([1.0, -0.5, 0.75, 1.25], dtype=torch.float32, device=input_ids.device).view(1, 1, 4)
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device).view(1, -1, 1) + 1.0
        return (x + pos * 0.2) * basis * self.weight


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_inputs(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    query_batches = [
        torch.tensor([[1, 3, 0], [2, 1, 4]], dtype=torch.long, device=device),
        torch.tensor([[2, 2, 1], [4, 1, 0]], dtype=torch.long, device=device),
    ]
    doc_batches = [
        torch.tensor([[3, 1, 0], [1, 4, 2]], dtype=torch.long, device=device),
        torch.tensor([[2, 3, 1], [4, 4, 1]], dtype=torch.long, device=device),
    ]
    return query_batches[rank], doc_batches[rank]


def _distributed_negative_inputs(rank: int, device: torch.device) -> torch.Tensor:
    neg_doc_batches = [
        torch.tensor(
            [
                [[4, 1, 0], [5, 1, 2]],
                [[2, 2, 1], [3, 4, 0]],
            ],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [
                [[5, 2, 1], [2, 4, 0]],
                [[1, 3, 2], [4, 2, 2]],
            ],
            dtype=torch.long,
            device=device,
        ),
    ]
    return neg_doc_batches[rank]


def _distributed_worker(rank: int, port: int, result_dir: str, loss_name: str, device_type: str) -> None:
    world_size = 2
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        standard_model = _ToyTokenModel().to(device)
        gradcache_model = _ToyTokenModel().to(device)
        gradcache_model.load_state_dict(standard_model.state_dict())

        query_ids, doc_ids = _distributed_inputs(rank, device)
        inputs = {
            "query_input_ids": query_ids,
            "query_attention_mask": (query_ids != 0).long(),
            "doc_input_ids": doc_ids,
            "doc_attention_mask": (doc_ids != 0).long(),
        }

        if loss_name == "pairwise":
            standard_loss_fn = ColbertPairwiseCELoss(temperature=1.0, normalize_scores=False)
            gradcache_loss_fn = GradCacheColbertPairwiseCELoss(
                mini_batch_size=1,
                temperature=1.0,
                normalize_scores=False,
            )
        elif loss_name == "infonce":
            standard_loss_fn = ColbertLoss(temperature=1.0, normalize_scores=False)
            gradcache_loss_fn = GradCacheColbertLoss(
                mini_batch_size=1,
                temperature=1.0,
                normalize_scores=False,
            )
        elif loss_name == "pairwise_negative":
            neg_doc_ids = _distributed_negative_inputs(rank, device)
            inputs["neg_doc_input_ids"] = neg_doc_ids
            inputs["neg_doc_attention_mask"] = (neg_doc_ids != 0).long()
            standard_loss_fn = ColbertPairwiseNegativeCELoss(
                temperature=1.0,
                normalize_scores=False,
                in_batch_term_weight=0.5,
            )
            gradcache_loss_fn = GradCacheColbertPairwiseNegativeCELoss(
                mini_batch_size=1,
                temperature=1.0,
                normalize_scores=False,
                in_batch_term_weight=0.5,
            )
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        query_embeddings = standard_model(query_ids)
        doc_embeddings = standard_model(doc_ids)
        gathered_doc_embeddings = torch.cat(all_gather(doc_embeddings), dim=0)
        if loss_name == "pairwise_negative":
            neg_doc_ids = inputs["neg_doc_input_ids"]
            neg_doc_embeddings = standard_model(neg_doc_ids.view(-1, neg_doc_ids.shape[-1])).view(
                neg_doc_ids.size(0), neg_doc_ids.size(1), neg_doc_ids.size(2), -1
            )
            standard_loss = standard_loss_fn(
                query_embeddings=query_embeddings,
                doc_embeddings=gathered_doc_embeddings,
                neg_doc_embeddings=neg_doc_embeddings,
                offset=rank * query_ids.size(0),
            )
        else:
            standard_loss = standard_loss_fn(
                query_embeddings=query_embeddings,
                doc_embeddings=gathered_doc_embeddings,
                offset=rank * query_ids.size(0),
            )
        standard_loss.backward()

        gradcache_loss = gradcache_loss_fn(gradcache_model, inputs)
        gradcache_loss.backward()

        result_path = Path(result_dir) / f"rank{rank}.pt"
        torch.save(
            {
                "standard_loss": standard_loss.detach().cpu(),
                "gradcache_loss": gradcache_loss.detach().cpu(),
                "standard_grad": standard_model.weight.grad.detach().cpu(),
                "gradcache_grad": gradcache_model.weight.grad.detach().cpu(),
            },
            result_path,
        )
    finally:
        dist.destroy_process_group()


def _run_distributed_equivalence(tmp_path: Path, loss_name: str, device_type: str) -> None:
    result_dir = tmp_path / f"{loss_name}-{device_type}"
    result_dir.mkdir()

    mp.spawn(
        _distributed_worker,
        args=(_free_port(), str(result_dir), loss_name, device_type),
        nprocs=2,
        join=True,
    )

    for rank in range(2):
        result = torch.load(result_dir / f"rank{rank}.pt")
        assert torch.allclose(result["standard_loss"], result["gradcache_loss"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(result["standard_grad"], result["gradcache_grad"], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("loss_name", ["pairwise", "infonce", "pairwise_negative"])
def test_gradcache_matches_distributed_full_batch_on_cpu(tmp_path, loss_name):
    _run_distributed_equivalence(tmp_path, loss_name=loss_name, device_type="cpu")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
@pytest.mark.parametrize("loss_name", ["pairwise", "infonce", "pairwise_negative"])
def test_gradcache_matches_distributed_full_batch_on_multi_gpu(tmp_path, loss_name):
    _run_distributed_equivalence(tmp_path, loss_name=loss_name, device_type="cuda")
