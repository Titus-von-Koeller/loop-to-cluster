"""fp32 single-GPU baseline: SmolLM2-135M trained from scratch on wikitext-2.

Stage 00 of a staged series. Each later stage (mixed precision, gradient
accumulation, DDP, FSDP) is a copy of this file with one change, diffed and
compared against it. Scheduling, clipping, evaluation and checkpointing are
absent by design: they are later stages, not oversights.
"""

import math

import torch
from datasets import load_dataset
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
torch.backends.cudnn.allow_tf32 = False  # on by default; fp32 would not be fp32

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:5%]")
rows = tokenizer(list(dataset["text"]), verbose=False)["input_ids"]
token_ids = torch.tensor([t for row in rows for t in row])

# One continuous stream cut into equal blocks: document boundaries are lost,
# which is what lets every batch be full and rectangular without padding.
num_blocks = token_ids.numel() // seq_len
blocks = token_ids[: num_blocks * seq_len].view(num_blocks, seq_len)

# A separate generator keeps shuffling independent of the RNG that weight
# initialisation consumed, so two runs of this file stay comparable.
generator = torch.Generator().manual_seed(0)
dataloader = DataLoader(
    blocks, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator
)


def cycle(loader):
    """Endless batches. Re-entering the loader reshuffles; itertools.cycle would not."""
    while True:
        yield from loader


batches = cycle(dataloader)
max_steps = num_epochs * len(dataloader)

cfg = AutoConfig.from_pretrained(model_name)
cfg.use_cache = False  # the KV cache only helps generation
cfg.dtype = torch.float32  # the checkpoint config says bfloat16; transformers 5 honors it

model = AutoModelForCausalLM.from_config(cfg)  # architecture only, weights random
model.to(device)
model.train()
assert next(model.parameters()).dtype == torch.float32, "baseline must be fp32"

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"parameters  {sum(p.numel() for p in model.parameters()):,}")
print(f"blocks      {num_blocks:,}")
print(f"max_steps   {max_steps}")
print(f"ln(V)       {math.log(cfg.vocab_size):.4f}   <- loss before learning anything")

for step in range(max_steps):
    ids = next(batches).to(device)          # data
    loss = model(ids, labels=ids).loss      # forward + score
    loss.backward()                         # backward
    optimizer.step()                        # update
    optimizer.zero_grad()                   # reset

    if step % log_every == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")