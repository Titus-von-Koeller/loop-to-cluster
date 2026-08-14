"""fp32 single-GPU baseline: SmolLM2-135M trained from scratch on wikitext-2."""

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

torch.manual_seed(0)

cfg = AutoConfig.from_pretrained(model_name)
cfg.dtype = torch.float32
cfg.use_cache = False

model = AutoModelForCausalLM.from_config(cfg)
model.to(device)
model.train()

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:5%]")
token_ids = torch.tensor(tokenizer("\n\n".join(dataset["text"]), verbose=False).input_ids)
num_blocks = token_ids.numel() // seq_len
blocks = token_ids[: num_blocks * seq_len].view(num_blocks, seq_len)
dataloader = DataLoader(blocks, batch_size=batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"parameters  {sum(p.numel() for p in model.parameters()):,}")
print(f"ln(V)       {math.log(cfg.vocab_size):.4f}")

history = []
for epoch in range(num_epochs):
    for step, ids in enumerate(dataloader):   # data
        ids = ids.to(device)
        loss = model(ids, labels=ids).loss    # forward + score
        loss.backward()                       # backward
        optimizer.step()                      # update
        optimizer.zero_grad()                 # reset

        history.append(loss.item())
        if step % 32 == 0:
            print(f"epoch {epoch}  step {step:3d}  {loss.item():.4f}")
