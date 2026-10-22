"""Sparse mixture-of-experts layers used by the REACT encoders.

The implementation follows the noisy top-k gating and FiLM-style expert
formulation used in the paper.  It is intentionally self-contained so that the
frame-level and video-level models share exactly the same routing code.
"""

from __future__ import annotations

import collections.abc
from itertools import repeat

import torch
from torch import nn
from torch.distributions.normal import Normal


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """Split an image-like feature map into non-overlapping patches."""

    def __init__(self, img_size=224, patch_size=16, in_chans=128, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input size {(height, width)} does not match the configured "
                f"size {self.img_size}."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class SparseDispatcher:
    """Route tokens to the experts selected by the gating network."""

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            device=stitched.device,
            dtype=stitched.dtype,
        )
        return zeros.index_add(0, self._batch_index, stitched)

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class FiLMExpert(nn.Module):
    """A lightweight feature-modulation expert: ``K(z) * z + B(z)``."""

    def __init__(self, dim):
        super().__init__()
        self.expert = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])

    def forward(self, x):
        modulation_product, modulation_add = self.expert
        return modulation_product(x) * x + modulation_add(x)


class MoE(nn.Module):
    """Noisy top-k mixture of FiLM experts with a load-balancing loss."""

    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        hidden_size,
        noisy_gating=True,
        k=4,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        self.experts = nn.ModuleList([FiLMExpert(input_size) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        if self.k > self.num_experts:
            raise ValueError("top-k cannot exceed the number of experts")

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        gates = torch.zeros_like(logits).scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(
                clean_logits, logits, noise_stddev, top_logits
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss
