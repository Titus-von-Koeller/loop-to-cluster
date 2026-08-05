"""The saved-tensor ledger: an inventory of what backward is actually holding.

"Activations" in a memory budget is usually a guess. This measures it. Autograd saves
every tensor it will need during the backward pass, and
`torch.autograd.graph.saved_tensors_hooks` fires once per save, so running a forward
pass under the hook yields the exact inventory — shapes, dtypes and bytes.

Entries are deduplicated by *storage*, not by tensor. One allocation saved by three
different ops is one entry, and a `.view()` of a tensor shares its storage so it is not
double counted. That matters immediately: transformers' loss path reshapes the logits
with `logits.view(-1, vocab_size)`, which would otherwise appear to double the largest
tensor in the model.

Each entry is classified:

    parameters    the storage belongs to a model parameter — a weight, not an
                  activation. In fp32 these are the weights autograd saved directly.
    weight_casts  a reduced-precision tensor whose shape matches a Linear weight.
                  Under autocast these are the cast cache: bf16 copies held alongside
                  the fp32 masters, which is why model states *grow* under mixed
                  precision instead of shrinking.
    logits        trailing dimension equal to vocab_size. Broken out because it is one
                  tensor big enough to dominate the total, and because the loss path
                  upcasts it — transformers' `ForCausalLMLoss` does `logits = logits.float()`
                  (loss/loss_utils.py:59) — so under autocast a bf16 *and* an fp32 copy
                  are both held. That single detail is why activations do not simply halve.
    activations   everything else.

**Scope.** This is an inventory of memory *held across* the backward pass, not a peak.
Tensors that backward allocates and frees transiently are invisible here, by design:
`l2c.harness.predict` refuses to predict them and `max_memory_allocated` catches them.

This pass never runs inside the timed loop. The hook is a Python callback per saved
tensor, so it would dominate step time — rule 2 in `l2c.harness.measure`.

In accelerate: no counterpart. accelerate configures mixed precision and leaves its
memory consequences implicit, which is exactly the gap this lab is built to close.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from l2c.harness.predict import autocast_eligible_weights

PARAMETERS = "parameters"
WEIGHT_CASTS = "weight_casts"
LOGITS = "logits"
ACTIVATIONS = "activations"
CATEGORIES = (PARAMETERS, WEIGHT_CASTS, LOGITS, ACTIVATIONS)


@dataclass(frozen=True, slots=True)
class SavedTensor:
    category: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(slots=True)
class Ledger:
    """Accumulates one entry per distinct storage saved for backward."""

    entries: list[SavedTensor] = field(default_factory=list)

    def bytes_in(self, *categories: str) -> int:
        wanted = set(categories) or set(CATEGORIES)
        return sum(e.nbytes for e in self.entries if e.category in wanted)

    def count_in(self, *categories: str) -> int:
        wanted = set(categories) or set(CATEGORIES)
        return sum(1 for e in self.entries if e.category in wanted)

    @property
    def total_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def by_category(self) -> dict[str, int]:
        return {name: self.bytes_in(name) for name in CATEGORIES}

    def by_dtype(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.entries:
            totals[entry.dtype] = totals.get(entry.dtype, 0) + entry.nbytes
        return dict(sorted(totals.items()))

    def summary(self) -> dict[str, object]:
        """A JSON-friendly digest, small enough to store with every run."""
        return {
            "total_bytes": self.total_bytes,
            "num_tensors": len(self.entries),
            "bytes_by_category": self.by_category(),
            "bytes_by_dtype": self.by_dtype(),
            "num_weight_cast_tensors": self.count_in(WEIGHT_CASTS),
        }


@contextmanager
def record(model: nn.Module, *, vocab_size: int) -> Iterator[Ledger]:
    """Record every tensor saved for backward by forwards run inside this block.

    Usage — the forward happens inside, the reporting after:

        with ledger.record(net, vocab_size=V) as inventory:
            loss = loss_from(net, batch)   # wrap in autocast for a mixed-precision arm
        del loss
        print(inventory.by_category())

    The pack and unpack hooks are the identity, so memory behavior is unchanged: this
    observes the graph without offloading any part of it.
    """
    inventory = Ledger()

    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    cast_shapes = {tuple(w.shape) for w in autocast_eligible_weights(model)}
    seen: set[int] = set()

    def classify(tensor: torch.Tensor, storage_ptr: int) -> str:
        if storage_ptr in param_storages:
            return PARAMETERS
        if tensor.dim() > 0 and tensor.shape[-1] == vocab_size:
            return LOGITS
        if tensor.dtype is not torch.float32 and tuple(tensor.shape) in cast_shapes:
            return WEIGHT_CASTS
        return ACTIVATIONS

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        if storage_ptr and storage_ptr not in seen:
            seen.add(storage_ptr)
            inventory.entries.append(
                SavedTensor(
                    category=classify(tensor, storage_ptr),
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).removeprefix("torch."),
                    nbytes=storage.nbytes(),
                )
            )
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield inventory
