# CLAUDE.md — loop-to-cluster

## Environment

Python 3.14 + PyTorch 2.13 (CUDA 13) managed by pixi. There is no system Python on this
host, so a bare `python3` fails.

**Name the project directory explicitly, in the same command:**

```bash
cd /home/titus/src/loop-to-cluster && pixi run python -c '...'
```

- **Never bare `pixi run`** from a parent directory: manifest lookup walks *upward* and
  will silently bind a different workspace rather than erroring. `--manifest-path <abs>`
  is the alternative when you cannot `cd`.
- **Do not rely on ambient direnv activation.** It is applied at shell-init, so
  `cd X && cmd` activates for the shell's *starting* directory, and an agent's cwd can
  reset between commands. Both fail silently with a working-but-wrong environment.
- The inline `cd` above is safe because `pixi run` resolves its manifest at exec time.
- Need `nvcc`, `CUDA_PATH` or nix's `libstdc++` as well? Use
  `direnv exec <abs-dir> <cmd>`, which supplies both layers from any cwd. Plain
  torch/CUDA work does not need it — the wheels bundle their CUDA userspace and
  `libcuda.so.1` comes from the system driver.

`.envrc` is untracked (machine-local) and carries a `PATH_add "$CONDA_PREFIX/bin"` line
after the pixi hook. That works around `use flake` clobbering the bin directory
`pixi shell-hook` prepends; see ADR 003 in `~/dotfiles/docs/adr/`. It is a convenience
only — the explicit invocations above are the dependable path.

## Jupyter notebooks

- Use the **Jupyter MCP** for all `.ipynb` operations: read, edit, insert, delete,
  execute.
- Do **not** use the built-in `NotebookEdit` tool. It writes cell source as a single JSON
  string, which produces whole-cell git diffs, and it edits the file behind JupyterLab's
  back — conflicting with the collaborative ydoc session the MCP server operates on.
- `.claude/settings.json` denies `NotebookEdit`, so this is enforced rather than merely
  requested. Configuration beats instruction: an instruction has to be noticed on every
  call, a deny rule does not.

### Running the stack

JupyterLab must be running for the MCP server to reach anything:

```bash
pixi run jupyter lab --no-browser --port 8888 \
  --ServerApp.ip 127.0.0.1 \
  --IdentityProvider.token "$(cat .jupyter-token)"
```

Bound to `127.0.0.1` on purpose. Upstream docs use `--ip 0.0.0.0`, which is only needed
when the MCP server runs in Docker; running it locally means Lab never has to listen on
the network.

`.jupyter-token` is a throwaway local token and is gitignored. The MCP server is
registered at **user** scope, so the token lives in `~/.claude.json` and is never
committed. If Lab restarts with a new token, re-register:

```bash
claude mcp remove jupyter
claude mcp add jupyter --scope user \
  -e JUPYTER_URL=http://127.0.0.1:8888 \
  -e JUPYTER_TOKEN="$(cat .jupyter-token)" \
  -- "$PWD/.pixi/envs/default/bin/jupyter-mcp-server" start --transport stdio
```

The MCP command points at the environment's binary directly rather than going through
`pixi run`, so nothing can write to stdout and corrupt the stdio protocol channel.

`DOCUMENT_ID` is deliberately unset: with it omitted, notebooks can be listed and chosen
per request. When set, it is a path relative to Lab's `root_dir`, which is this
directory.

### Corrections to older setup guides

Verified on 2026-08-04 against `jupyter-mcp-server` 1.2.0:

- **A subcommand is required**: `start --transport stdio`. A bare `jupyter-mcp-server`
  prints help and exits, so it never starts a server.
- **Do not swap `pycrdt` for `datalayer_pycrdt`.** That step is stale and actively
  harmful: `datalayer_pycrdt` is frozen at 0.12.17 while `jupyter-nbmodel-client`
  requires `pycrdt>=0.12.50`. Datalayer's own current quick start installs upstream
  `pycrdt`.
- **Upstream version pins are stale** (jupyterlab 4.4.1, jupyter-collaboration 4.0.2).
  Unpinned resolves to 4.6.2 / 5.0.0, which works — `jupyter-mcp-server` itself depends
  on `jupyter-collaboration` unpinned.
- **`ALLOW_IMG_OUTPUT` is not in the 1.2 interface.** Setting it does nothing.
- `jupyter-collaboration` does not need installing separately; `jupyter-mcp-server`
  depends on it.

To confirm the collaboration extension is live, check that
`/api/collaboration/session/x` returns **405** (route registered, wrong method) rather
than 404.
