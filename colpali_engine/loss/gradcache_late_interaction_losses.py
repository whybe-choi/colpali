from functools import partial

import torch
import torch.nn as nn
import tqdm
from torch.distributed.nn.functional import all_gather
from torch.utils.checkpoint import get_device_states, set_device_states

from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
    ColbertPairwiseCELoss,
    ColbertPairwiseNegativeCELoss,
)


class RandContext:
    """
    Random-state context manager that captures both CPU and GPU random states.
    This ensures that when re‑executing a forward pass (e.g. in GradCache’s second pass),
    stochastic operations produce identical outputs.
    """
    def __init__(self, *tensors) -> None:
        # Capture CPU RNG state.
        self.fwd_cpu_state = torch.get_rng_state()
        # Capture GPU states for all devices associated with the provided tensors.
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        # Fork the RNG states on the captured devices.
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        # Reset the CPU RNG state.
        torch.set_rng_state(self.fwd_cpu_state)
        # Reset the GPU RNG states.
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _gather_doc_embeddings(doc_embeddings: torch.Tensor, local_batch_size: int) -> tuple[torch.Tensor, int]:
    if not _is_distributed():
        return doc_embeddings, 0

    gathered_doc_embeddings = torch.cat(all_gather(doc_embeddings), dim=0)
    offset = torch.distributed.get_rank() * local_batch_size
    return gathered_doc_embeddings, offset


def _flatten_negative_inputs(neg_doc_features: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    if not neg_doc_features:
        return neg_doc_features, 0

    input_ids = neg_doc_features["input_ids"]
    num_neg_docs = input_ids.size(1)
    flattened_features = {key: value.reshape(-1, *value.shape[2:]) for key, value in neg_doc_features.items()}
    return flattened_features, num_neg_docs


def _reshape_negative_embeddings(neg_doc_embeddings: torch.Tensor, num_neg_docs: int) -> torch.Tensor:
    if num_neg_docs <= 0:
        raise ValueError("num_neg_docs must be positive when reshaping negative embeddings")
    return neg_doc_embeddings.reshape(-1, num_neg_docs, *neg_doc_embeddings.shape[1:])

def _backward_hook(grad_output, sentence_features, random_states, loss_obj, model):
    """
    Backward hook that re-computes the embeddings in mini-batches with gradients enabled
    and uses the cached gradients to backpropagate. This version wraps the forward pass in the
    corresponding RandContext to reproduce the same randomness.
    """
    mini_batch_size = loss_obj.mini_batch_size
    # sentence_features: a list with two dicts [query_features, doc_features]
    # random_states: a list with one RandContext list per feature branch.
    assert loss_obj.cache is not None
    assert random_states is not None
    with torch.enable_grad():
        for branch_feature, branch_cache, branch_random_states in zip(sentence_features, loss_obj.cache, random_states):
            input_ids = branch_feature["input_ids"]
            bsz = input_ids.size(0)
            # Iterate over mini-batches.
            for idx, start in enumerate(range(0, bsz, mini_batch_size)):
                end = start + mini_batch_size
                mini_feature = {k: v[start:end] for k, v in branch_feature.items()}
                # Use the stored RandContext if available.
                r_state = branch_random_states[idx]
                if r_state is not None:
                    with r_state:
                        mini_embeds = model.forward(**mini_feature)
                else:
                    mini_embeds = model.forward(**mini_feature)
                cached_grad = branch_cache[idx]
                # Compute a surrogate loss that replays the cached gradient.
                surrogate = torch.dot(mini_embeds.flatten(), cached_grad.flatten()) * grad_output
                surrogate.backward()


class GradCacheColbertLoss(nn.Module):
    def __init__(
        self,
        mini_batch_size: int = 32,
        scale: float = 1.0,
        show_progress_bar: bool = False,
        temperature: float | None = None,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        """
        GradCache enabled version of the ColBERT loss.

        Args:
            mini_batch_size: Number of items per mini-batch.
            scale: Scaling factor for the similarity scores.
            show_progress_bar: If True, shows progress bars during mini-batch processing.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.cache = None
        self.random_states = None
        self.show_progress_bar = show_progress_bar
        self.gradcache_enabled = True  # Flag indicating GradCache is active.
        self.inner_loss = ColbertLoss(
            temperature=1 / scale if temperature is None else temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def embed_minibatch_iter(self, model, sentence_feature: dict, with_grad: bool, copy_random_state: bool):
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)
        for start in tqdm.trange(0, bsz, self.mini_batch_size, desc="Embedding minibatches",
                                 disable=not self.show_progress_bar):
            end = start + self.mini_batch_size
            mini_feature = {k: v[start:end] for k, v in sentence_feature.items()}
            random_state = None
            if copy_random_state:
                random_state = RandContext(*mini_feature.values())
            grad_context = torch.enable_grad() if with_grad else torch.no_grad()
            with grad_context:
                mini_embeds = model.forward(**mini_feature)
                mini_embeds = mini_embeds.detach().requires_grad_(True)
            yield mini_embeds, random_state

    def calculate_loss(self, reps: list[list[torch.Tensor]], with_backward: bool = False) -> torch.Tensor:
        embeddings_query = torch.cat(reps[0], dim=0)  # shape: (total_query, seq_len, dim)
        embeddings_doc = torch.cat(reps[1], dim=0)  # shape: (local_doc_batch, seq_len, dim)
        gathered_doc_embeddings, offset = _gather_doc_embeddings(
            embeddings_doc,
            local_batch_size=embeddings_query.size(0),
        )
        loss = self.inner_loss(
            query_embeddings=embeddings_query,
            doc_embeddings=gathered_doc_embeddings,
            offset=offset,
        )
        if with_backward:
            loss.backward()
        return loss

    def calculate_loss_and_cache_gradients(self, reps: list[list[torch.Tensor]]) -> torch.Tensor:
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()
        # Cache gradients for each mini-batch.
        self.cache = []
        for branch in reps:
            branch_cache = []
            for r in branch:
                branch_cache.append(r.grad)
            self.cache.append(branch_cache)
        return loss

    def forward(self, model, inputs: dict) -> torch.Tensor:
        """
        inputs: dict containing keys with prefixes "query_" and "doc_".
        """
        # Remove prefixes.
        query_features = {k.replace("query_", ""): v for k, v in inputs.items() if k.startswith("query_")}
        doc_features = {k.replace("doc_", ""): v for k, v in inputs.items() if k.startswith("doc_")}

        # === First Pass: Get embeddings without gradients, capturing RandContext.
        reps_query = []
        rs_query = []
        for mini_embeds, rs in self.embed_minibatch_iter(model, query_features, with_grad=False,
                                                         copy_random_state=True):
            reps_query.append(mini_embeds)
            rs_query.append(rs)
        reps_doc = []
        rs_doc = []
        for mini_embeds, rs in self.embed_minibatch_iter(model, doc_features, with_grad=False, copy_random_state=True):
            reps_doc.append(mini_embeds)
            rs_doc.append(rs)
        reps = [reps_query, reps_doc]
        self.random_states = [rs_query, rs_doc]

        if torch.is_grad_enabled():
            # Step (2): Compute loss and cache gradients.
            loss = self.calculate_loss_and_cache_gradients(reps)
            # Step (3): Re-run embeddings with gradients enabled and register a hook that uses the cached gradients.
            loss.register_hook(partial(_backward_hook, sentence_features=[query_features, doc_features],
                                       random_states=self.random_states, loss_obj=self, model=model))
        else:
            loss = self.calculate_loss(reps, with_backward=False)
        return loss
class GradCacheColbertPairwiseCELoss(nn.Module):
    def __init__(
        self,
        mini_batch_size: int = 32,
        scale: float = 1.0,
        show_progress_bar: bool = False,
        temperature: float | None = None,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        """
        GradCache-enabled version of the ColBERTPairwiseCELoss.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.cache = None
        self.random_states = None
        self.show_progress_bar = show_progress_bar
        self.gradcache_enabled = True
        self.inner_loss = ColbertPairwiseCELoss(
            temperature=1 / scale if temperature is None else temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def embed_minibatch_iter(self, model, sentence_feature: dict, with_grad: bool, copy_random_state: bool):
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)
        for start in tqdm.trange(0, bsz, self.mini_batch_size, desc="Embedding minibatches",
                                 disable=not self.show_progress_bar):
            end = start + self.mini_batch_size
            mini_feature = {k: v[start:end] for k, v in sentence_feature.items()}
            random_state = RandContext(*mini_feature.values()) if copy_random_state else None
            grad_context = torch.enable_grad() if with_grad else torch.no_grad()
            with grad_context:
                mini_embeds = model.forward(**mini_feature)
                mini_embeds = mini_embeds.detach().requires_grad_(True)
            yield mini_embeds, random_state

    def calculate_loss(self, reps: list[list[torch.Tensor]], with_backward: bool = False) -> torch.Tensor:
        embeddings_query = torch.cat(reps[0], dim=0)
        embeddings_doc = torch.cat(reps[1], dim=0)
        gathered_doc_embeddings, offset = _gather_doc_embeddings(
            embeddings_doc,
            local_batch_size=embeddings_query.size(0),
        )
        loss = self.inner_loss(
            query_embeddings=embeddings_query,
            doc_embeddings=gathered_doc_embeddings,
            offset=offset,
        )
        if with_backward:
            loss.backward()
        return loss

    def calculate_loss_and_cache_gradients(self, reps: list[list[torch.Tensor]]) -> torch.Tensor:
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = []
        for branch in reps:
            branch_cache = [r.grad for r in branch]
            self.cache.append(branch_cache)
        return loss

    def forward(self, model, inputs: dict) -> torch.Tensor:
        # Remove prefixes.
        query_features = {k.replace("query_", ""): v for k, v in inputs.items() if k.startswith("query_")}
        doc_features = {k.replace("doc_", ""): v for k, v in inputs.items() if k.startswith("doc_")}

        # First pass: get embeddings without gradients (and capture RandContext).
        reps_query, rs_query = [], []
        for mini_embeds, rs in self.embed_minibatch_iter(model, query_features, with_grad=False,
                                                         copy_random_state=True):
            reps_query.append(mini_embeds)
            rs_query.append(rs)
        reps_doc, rs_doc = [], []
        for mini_embeds, rs in self.embed_minibatch_iter(model, doc_features, with_grad=False, copy_random_state=True):
            reps_doc.append(mini_embeds)
            rs_doc.append(rs)
        reps = [reps_query, reps_doc]
        self.random_states = [rs_query, rs_doc]

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(
                partial(
                    _backward_hook,
                    sentence_features=[query_features, doc_features],
                    random_states=self.random_states,
                    loss_obj=self,
                    model=model,
                )
            )
        else:
            loss = self.calculate_loss(reps, with_backward=False)
        return loss

class GradCacheColbertPairwiseNegativeCELoss(nn.Module):
    def __init__(
        self,
        mini_batch_size: int = 32,
        in_batch_term: bool = False,
        show_progress_bar: bool = False,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: float | None = None,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        """
        GradCache-enabled version of the ColBERTPairwiseNegativeCELoss.

        Args:
            in_batch_term: If True, includes an additional in-batch loss term.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.cache = None
        self.random_states = None
        self.show_progress_bar = show_progress_bar
        self.gradcache_enabled = True
        self.num_neg_docs = 0
        self.inner_loss = ColbertPairwiseNegativeCELoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            in_batch_term_weight=0.5 if in_batch_term_weight is None and in_batch_term else (
                0.0 if in_batch_term_weight is None else in_batch_term_weight
            ),
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def embed_minibatch_iter(self, model, sentence_feature: dict, with_grad: bool, copy_random_state: bool):
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)
        for start in tqdm.trange(0, bsz, self.mini_batch_size, desc="Embedding minibatches",
                                 disable=not self.show_progress_bar):
            end = start + self.mini_batch_size
            mini_feature = {k: v[start:end] for k, v in sentence_feature.items()}
            random_state = RandContext(*mini_feature.values()) if copy_random_state else None
            grad_context = torch.enable_grad() if with_grad else torch.no_grad()
            with grad_context:
                mini_embeds = model.forward(**mini_feature)
                mini_embeds = mini_embeds.detach().requires_grad_(True)
            yield mini_embeds, random_state

    def calculate_loss(self, reps: list[list[torch.Tensor]], with_backward: bool = False) -> torch.Tensor:
        embeddings_query = torch.cat(reps[0], dim=0)  # (batch, num_query_tokens, dim)
        embeddings_doc = torch.cat(reps[1], dim=0)  # (local batch, num_doc_tokens, dim)
        embeddings_neg_doc = _reshape_negative_embeddings(
            torch.cat(reps[2], dim=0),
            num_neg_docs=self.num_neg_docs,
        )
        gathered_doc_embeddings, offset = _gather_doc_embeddings(
            embeddings_doc,
            local_batch_size=embeddings_query.size(0),
        )
        loss = self.inner_loss(
            query_embeddings=embeddings_query,
            doc_embeddings=gathered_doc_embeddings,
            neg_doc_embeddings=embeddings_neg_doc,
            offset=offset,
        )
        if with_backward:
            loss.backward()
        return loss

    def calculate_loss_and_cache_gradients(self, reps: list[list[torch.Tensor]]) -> torch.Tensor:
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = []
        for branch in reps:
            branch_cache = [r.grad for r in branch]
            self.cache.append(branch_cache)
        return loss

    def forward(self, model, inputs: dict) -> torch.Tensor:
        # Remove prefixes.
        query_features = {k.replace("query_", ""): v for k, v in inputs.items() if k.startswith("query_")}
        doc_features = {k.replace("doc_", ""): v for k, v in inputs.items() if k.startswith("doc_")}
        neg_doc_features = {k.replace("neg_doc_", ""): v for k, v in inputs.items() if k.startswith("neg_doc_")}
        neg_doc_features, self.num_neg_docs = _flatten_negative_inputs(neg_doc_features)

        # First pass: get embeddings without gradients and capture RandContext.
        reps_query, rs_query = [], []
        for mini_embeds, rs in self.embed_minibatch_iter(model, query_features, with_grad=False,
                                                         copy_random_state=True):
            reps_query.append(mini_embeds)
            rs_query.append(rs)
        reps_doc, rs_doc = [], []
        for mini_embeds, rs in self.embed_minibatch_iter(model, doc_features, with_grad=False, copy_random_state=True):
            reps_doc.append(mini_embeds)
            rs_doc.append(rs)
        reps_neg_doc, rs_neg_doc = [], []
        for mini_embeds, rs in self.embed_minibatch_iter(model, neg_doc_features, with_grad=False,
                                                         copy_random_state=True):
            reps_neg_doc.append(mini_embeds)
            rs_neg_doc.append(rs)
        reps = [reps_query, reps_doc, reps_neg_doc]
        self.random_states = [rs_query, rs_doc, rs_neg_doc]

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(partial(_backward_hook,
                                       sentence_features=[query_features, doc_features, neg_doc_features],
                                       random_states=self.random_states,
                                       loss_obj=self, model=model))
        else:
            loss = self.calculate_loss(reps, with_backward=False)
        return loss
