"""A small pre-LN decoder-only transformer.

Deliberately plain: no dropout, no bias in the projections, learned position
embeddings, untied `lm_head`. Every allocation in the forward pass is meant to be
predictable on paper, which is what makes the measured saved-tensor ledger in
`instrument.py` comparable against the analytic one in `predict.py`.

Initialised from a CPU generator so parameters are identical across precision modes
and devices for a given seed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # One matmul for q, k, v. The split/view/transpose below are all views, so the
        # single (B, S, 3D) activation is what backward holds on to.
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # reshape() after transpose() cannot alias, so this copy is the proj input.
        y = y.transpose(1, 2).reshape(B, S, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def init_weights(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        for name, p in self.named_parameters():
            with torch.no_grad():
                if p.dim() >= 2:
                    std = 0.02
                    # GPT-2's residual-path scaling: keeps activation variance from
                    # growing with depth, so the loss curve does not depend on n_layer
                    # in a way that would confound depth sweeps.
                    if name.endswith(("proj.weight", "fc2.weight")):
                        std /= math.sqrt(2 * self.cfg.n_layer)
                    p.copy_(torch.empty(p.shape).normal_(0.0, std, generator=g))
                else:
                    p.fill_(1.0 if name.endswith("weight") else 0.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        return F.cross_entropy(logits.view(B * S, -1), targets.reshape(B * S))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def matmul_param_count(self) -> int:
        """Parameters autocast casts to low precision, i.e. `nn.Linear` weights.

        Embedding and LayerNorm weights are not autocast-eligible, so they never get a
        cached low-precision copy. This is the base of the weight-cache prediction.
        """
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))
