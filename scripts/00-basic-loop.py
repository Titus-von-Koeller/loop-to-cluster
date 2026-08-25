"""An fp32 training loop on one GPU: SmolLM2-135M trained from scratch on wikitext-2.

First in a numbered series of scripts. Each later one is a copy of this file with one
technique added — mixed precision, say, or gradient accumulation — so this file is both the
thing to read first and the run the others are compared against. How that comparison works
is in DECISIONS.md.

Read it top to bottom: the corpus is tokenized and packed into fixed-length blocks, a model
is built from SmolLM2's architecture with random weights, and the loop at the bottom does
the four things a training step does — forward, score, backward, update.

What it establishes is that the mechanics run: the loss starts near ln(V), which is what a
model that knows nothing scores, and falls from there. What it does not establish is that
training works. There is no held-out split, so a falling loss cannot separate learning from
memorizing over three passes of a small corpus.

Left out on purpose: learning-rate schedule and warmup, gradient clipping, evaluation,
checkpointing. Each arrives with the topic that studies it. The metrics below are logged
because real training code logs them; recording where memory goes lives outside the script,
in snapshot.py, so that every topic is measured the same way.
"""

import math
import time

import torch
import trackio
from datasets import load_dataset
from torch.nn.utils import get_total_norm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-135M"
device = "cuda:0"
num_epochs = 3
batch_size = 8
seq_len = 128
learning_rate = 1e-4
log_every = 32

torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")  # "high" would truncate matmuls to TF32

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:5%]")
rows = tokenizer(list(dataset["text"]), verbose=False)["input_ids"]
token_ids = torch.tensor([t for row in rows for t in row])

# One continuous stream cut into equal blocks: document boundaries are lost, which
# is what lets every batch be full and rectangular without padding. Equal token
# counts per batch are also what let a later script's average match this one exactly.
num_blocks = token_ids.numel() // seq_len
blocks = token_ids[: num_blocks * seq_len].view(num_blocks, seq_len)

# A separate generator keeps shuffling independent of the RNG that weight
# initialization consumed, so runs stay comparable despite different model cfg.
generator = torch.Generator().manual_seed(0)
dataloader = DataLoader(blocks, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)


def cycle(loader):
    """Endless batches. Re-entering the loader reshuffles; itertools.cycle would not."""
    while True:
        yield from loader


batches = cycle(dataloader)
max_steps = num_epochs * len(dataloader)

cfg = AutoConfig.from_pretrained(model_name)
cfg.use_cache = False  # the KV cache only helps generation
cfg.dtype = torch.float32  # the checkpoint config says bfloat16

model = AutoModelForCausalLM.from_config(cfg)  # architecture only, weights random
model.to(device)
model.train()
assert next(model.parameters()).dtype == torch.float32, "baseline must be fp32"

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"parameters  {sum(p.numel() for p in model.parameters()):,}")
print(f"blocks      {num_blocks:,}")
print(f"max_steps   {max_steps}")
print(f"ln(V)       {math.log(cfg.vocab_size):.4f}   <- loss before learning anything")

trackio.init(
    project="loop-to-cluster",
    name="00-basic-loop",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "dtype": "float32",
    },
)

for step in range(max_steps):
    started = time.perf_counter()

    ids = next(batches).to(device)  # data
    loss = model(ids, labels=ids).loss  # forward + score
    loss.backward()  # backward
    # read here: the update does not change it, but zero_grad erases it
    grad_norm = get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
    optimizer.step()  # update
    optimizer.zero_grad()  # reset

    # CUDA is asynchronous, so without this the timer would stop when the last kernel was
    # *queued* rather than when it finished, and every tokens/s number would be too high.
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    trackio.log(
        {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "step_time_s": elapsed,
            "tokens_per_s": ids.numel() / elapsed,
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        },
        step=step,
    )
    # Reset so the next step's peak is that step's own. Left running, this is a run-wide
    # high-water mark that stops moving after the first update and logs a flat line.
    torch.cuda.reset_peak_memory_stats(device)

    if step % log_every == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}  grad_norm {grad_norm.item():.3f}")

trackio.finish()
