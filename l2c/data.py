"""Synthetic token data.

A fixed corpus of random tokens, drawn once from a CPU generator and held on the GPU.
There is no dataloader and no host-to-device copy inside the step, so step time measures
the model and the optimiser rather than the input pipeline.

The task is unlearnable by construction — tokens are i.i.d. uniform — so the loss falls
from log(V) only as far as the model can memorise the corpus. That is fine: the question
is whether two precision modes trace the *same* curve, not whether the curve is
interesting.
"""

from __future__ import annotations

import torch

from .config import DataConfig


class SyntheticCorpus:
    def __init__(self, cfg: DataConfig, vocab_size: int, device: torch.device) -> None:
        g = torch.Generator().manual_seed(cfg.data_seed)
        tokens = torch.randint(
            0, vocab_size, (cfg.corpus_tokens,), generator=g, dtype=torch.int64
        )
        self.tokens = tokens.to(device)
        self.cfg = cfg
        self.batch_tokens = cfg.batch_size * cfg.seq_len
        self.n_batches = (len(self.tokens) - 1) // self.batch_tokens

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch `step`, wrapping around the corpus. Deterministic in `step` alone.

        Cloned rather than handed back as a slice view. A view's storage is the entire
        corpus, so backward saving the index tensor would charge all 16 MiB of corpus to
        the activation ledger; a clone owns exactly B*S*8 bytes, which is also what a
        real dataloader would hand over.
        """
        start = (step % self.n_batches) * self.batch_tokens
        end = start + self.batch_tokens
        shape = (self.cfg.batch_size, self.cfg.seq_len)
        x = self.tokens[start:end].view(shape).clone()
        y = self.tokens[start + 1 : end + 1].view(shape).clone()
        return x, y

    def nbytes(self) -> int:
        return self.tokens.numel() * self.tokens.element_size()
