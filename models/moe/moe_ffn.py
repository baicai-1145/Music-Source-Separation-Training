import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Simple two-layer expert MLP mirroring the dense FFN structure."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoEFeedForward(nn.Module):
    """
    Mixture-of-Experts feed-forward layer with top-k routing and load-balancing loss.

    The design is inspired by Qwen3Next's high-sparsity MoE: we maintain many experts but only
    activate a small subset per token, which keeps the runtime cost close to the dense baseline
    while increasing the effective model capacity.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 32,
        top_k: int = 2,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
        capacity_factor: Optional[float] = 1.1,
    ):
        super().__init__()
        assert num_experts >= top_k >= 1, "num_experts must be >= top_k >= 1"
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            ExpertMLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor with shape (batch, seq, dim)
        Returns:
            output tensor of the same shape as x and the auxiliary load balancing loss.
        """
        b, n, d = x.shape
        tokens = b * n
        if tokens == 0:
            return x, x.new_zeros([])

        flat_x = x.reshape(tokens, d)

        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_vals, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        capacity = None
        if self.capacity_factor is not None and self.capacity_factor > 0:
            capacity = max(
                1,
                int(
                    math.ceil(
                        self.capacity_factor
                        * (self.top_k * tokens)
                        / self.num_experts
                    )
                ),
            )

        output = flat_x.new_zeros(tokens, d)

        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id  # (tokens, top_k)
            if not mask.any():
                continue

            positions = mask.nonzero(as_tuple=False)
            if capacity is not None and positions.shape[0] > capacity:
                gating_scores = topk_vals[positions[:, 0], positions[:, 1]]
                top_gating, keep_indices = torch.topk(
                    gating_scores, k=capacity, dim=0
                )
                positions = positions[keep_indices]
                gating_scores = top_gating
            else:
                gating_scores = topk_vals[positions[:, 0], positions[:, 1]]

            token_indices = positions[:, 0]

            expert_input = flat_x[token_indices]
            expert_output = expert(expert_input)

            weighted_output = expert_output * gating_scores.unsqueeze(-1)
            weighted_output = weighted_output.to(output.dtype)
            output[token_indices] += weighted_output

        aux_loss = self._load_balancing_loss(router_probs, topk_idx)
        out = output.view(b, n, d)
        return out, aux_loss

    def _load_balancing_loss(
        self, router_probs: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        if self.aux_loss_weight <= 0:
            return router_probs.new_zeros([])

        # (tokens, top_k, num_experts)
        expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.mean(dim=0)  # (top_k, num_experts)
        router_prob_per_expert = router_probs.mean(dim=0)  # (num_experts,)
        balancing = (
            tokens_per_expert * router_prob_per_expert.unsqueeze(0)
        ).sum() * self.num_experts
        return balancing * self.aux_loss_weight
