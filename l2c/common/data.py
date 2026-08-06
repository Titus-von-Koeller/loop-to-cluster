"""Data. Shared, because no lesson is about tokenization.

Tokenize once, cache to a `.pt` file, and every later run is offline and instant.

**Packed, fixed-length blocks.** The corpus is concatenated into one token stream and
cut into `seq_len` chunks, so every sample holds exactly the same number of tokens.
No padding, no attention mask, no wasted compute. That uniformity is what keeps the
memory arithmetic clean for steps 1 and 2: activation memory is a function of
`batch_size * seq_len` and nothing else.

**Why that matters later.** At step 3 the format deliberately changes to *padded,
variable-length* batches. Dividing an accumulated loss by `grad_accum_steps` is
exactly right only when every microbatch contributes the same number of tokens; with
uneven batches the correct weight is the token count. Fixed-length data makes that
bug structurally invisible, so the data format change is what makes the lesson
appear. accelerate carries the machinery for the general case:
`GradientState.remainder` and `end_of_dataloader` (accelerate/state.py) exist so that
`Accelerator.accumulate` can tell a full microbatch from a short final one.

In accelerate, the dataloader is the most heavily rewritten object of the four that
`prepare()` accepts. Worth knowing by name before step 4:

- `SeedableRandomSampler` (data_loader.py:73) — the reason `batch_generator` below
  exists. A plain `RandomSampler` draws from the global RNG, so its order depends on
  how many draws happened earlier in the process; that breaks reproducibility and
  resumption. accelerate replaces it with a sampler holding its own seed.
- `DataLoaderShard` (data_loader.py:510) — each rank loads the whole dataset and
  takes every Nth batch.
- `DataLoaderDispatcher` (data_loader.py:722) — rank 0 loads and scatters batches to
  the others. The `dispatch_batches` choice between these two is a step-4 topic.
- `BatchSamplerShard` (110), `IterableDatasetShard` (274) — the index-level and
  stream-level equivalents.
"""

from collections.abc import Iterable, Iterator

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from l2c.paths import cache_dir

#: Bumped when the packing changes, so a stale cache cannot be silently reused.
_CACHE_VERSION = 1


def batch_generator(seed: int) -> torch.Generator:
    """A generator dedicated to batch order.

    Passed to `DataLoader(generator=...)` so that shuffling draws from here and not
    from the global RNG. Without it, batch order depends on how many random numbers
    model initialization happened to consume, which means changing the architecture
    silently changes which data the model sees — and two runs you intended to compare
    are no longer comparable.

    This is the same problem `accelerate.data_loader.SeedableRandomSampler` solves.
    """
    return torch.Generator().manual_seed(seed)


def endless[T](loader: Iterable[T]) -> Iterator[T]:
    """Yield batches forever, restarting the loader at each epoch boundary.

    Steps are counted in optimizer updates, not epochs, so the loop should not have to
    know where an epoch ends. Restarting the loader redraws the shuffle from the
    generator it was built with, so the sequence stays deterministic across the wrap.
    """
    while True:
        yield from loader


def build_packed_dataset(
    tokenizer_name: str,
    *,
    seq_len: int = 512,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    max_blocks: int | None = 4096,
) -> TensorDataset:
    """A `TensorDataset` of `(input_ids,)` blocks, each exactly `seq_len` long.

    Labels are the inputs; `LlamaForCausalLM` shifts them internally.

    `Salesforce/wikitext` is the namespaced id. The bare `wikitext` name relied on a
    loading script, and `datasets` 3.0 removed script-based loading.
    """
    tag = f"v{_CACHE_VERSION}-{dataset_config}-{split}-{seq_len}-{max_blocks}"
    cache = cache_dir() / f"{tag}.pt"

    if cache.exists():
        blocks = torch.load(cache, weights_only=True)
        return TensorDataset(blocks)

    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    raw = load_dataset(dataset_name, dataset_config, split=split)

    # Column access, not row iteration: `raw["text"]` is one arrow read, whereas
    # iterating rows pays Python overhead per row.
    texts = [text for text in raw["text"] if text.strip()]
    encoded = tokenizer(texts)["input_ids"]

    # An EOS between documents keeps a block from silently welding the end of one
    # article to the start of the next. It costs one token per document.
    stream: list[int] = []
    for sequence in encoded:
        stream.extend(sequence)
        stream.append(tokenizer.eos_token_id)

    n_blocks = len(stream) // seq_len
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)
    blocks = torch.tensor(stream[: n_blocks * seq_len], dtype=torch.long)
    blocks = blocks.view(n_blocks, seq_len)

    torch.save(blocks, cache)
    return TensorDataset(blocks)
