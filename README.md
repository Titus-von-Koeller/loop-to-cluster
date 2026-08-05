# loop-to-cluster

PyTorch workspace managed with [pixi](https://pixi.sh).

## Environment

| | |
| --- | --- |
| Python | 3.14 |
| PyTorch | 2.13, CUDA 13 wheels from PyPI |

Python 3.14 is the newest version with full support across the training stack: torch
ships `cp314` wheels, triton has `cp314` and caps itself at `<3.15`, numpy has `cp314`,
and `tokenizers`/`safetensors` publish `abi3` wheels so they are version-agnostic.

## Usage

```bash
pixi install                                              # materialize the environment
pixi run python -c 'import torch; print(torch.cuda.is_available())'
```

`direnv` activates the environment on `cd` via `.envrc`, which is intentionally not
committed — it is machine-local. It layers this project's pixi environment on top of the
CUDA and native toolchain from the nix devShell at `~/src`.

## Notes

- `torch` comes from PyPI rather than conda-forge. conda-forge builds trail for
  brand-new CPython versions, and since 2.11 the default PyPI wheels *are* the CUDA 13
  build — torch 2.13.0 depends on `nvidia-cudnn-cu13`, `nvidia-nccl-cu13` and friends —
  so no `download.pytorch.org` index is required. The tradeoff is that anything
  depending on torch must also come from PyPI, not conda-forge.
- `libcuda.so.1` comes from the host driver, not from the environment: no wheel can
  bundle it, since it has to match the loaded kernel module.
- `pixi.toml` and `pixi.lock` are committed here on purpose. A global gitignore keeps
  them untracked in cloned upstream repos; `.gitignore` negates that for this project,
  where they are the environment definition.
