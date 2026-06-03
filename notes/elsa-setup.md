# ELSA setup (one-time)

Step-by-step to get `pokemon-rl` running on TCNJ's ELSA cluster. Do this
once; future training runs are then just `sbatch` away.

## 0. Prerequisites
- TCNJ VPN if off-campus: `vpn.tcnj.edu`
- ELSA account: `guenthc1@elsa.hpc.tcnj.edu`
- Advisor cleared for personal-project use; escalation contact: Sean Sivy

## 1. SSH in
```sh
ssh guenthc1@elsa.hpc.tcnj.edu
```

## 2. Install `uv` if missing
```sh
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Add the PATH line to `~/.bashrc` so future logins find uv.

## 3. Clone the repo into scratch
ELSA's home dir is small. All build / data / runs live in `/scratch`.

```sh
mkdir -p /scratch/guenthc1
cd /scratch/guenthc1
git clone https://github.com/colingoonth/pokemon-rl.git
cd pokemon-rl
```

## 4. Install deps
```sh
uv sync
```

## 5. Upload ROM + save state (from local Macbook, not ELSA)
The ROM and save state are gitignored — copy them up manually.

From local:
```sh
scp ~/pokemon-rl/roms/pokemon_red.gb \
    guenthc1@elsa.hpc.tcnj.edu:/scratch/guenthc1/pokemon-rl/roms/pokemon_red.gb

scp ~/pokemon-rl/states/post_intro.state \
    guenthc1@elsa.hpc.tcnj.edu:/scratch/guenthc1/pokemon-rl/states/post_intro.state
```

Verify on ELSA:
```sh
cd /scratch/guenthc1/pokemon-rl
shasum roms/pokemon_red.gb
# expected: ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

## 6. Hello-world SLURM job (smoke test the whole pipeline)
```sh
cd /scratch/guenthc1/pokemon-rl
sbatch slurm/hello_gpu.sbatch
```

`sbatch` prints a job id. Check status:
```sh
squeue -u guenthc1
```

When done, read the output:
```sh
cat /scratch/guenthc1/pokemon-rl/runs/hello_<JOB_ID>/hello.out
```

Should show GPU name (L40S), CUDA available True, and a sum of 1048576
from the smoke tensor. If yes — the cluster + uv + torch + CUDA all
work end-to-end.

## 7. Sanity-run tests
```sh
uv run pytest tests/ -v
```
Both RAM-map and integration tests should pass (skips network-cluster-only).

## 8. (Later) Submit real training
```sh
sbatch slurm/train_brock.sbatch
```
Outputs land in `runs/brock_<JOB_ID>/`. Sync to local with
`./scripts/sync_from_elsa.sh brock_<JOB_ID>` (run from local Macbook).

## Troubleshooting

- **`nvidia-smi: command not found` on login node** — expected. GPUs live on
  compute nodes; only the SLURM job sees them.
- **Job stuck in PD (pending)** — `squeue -u guenthc1` shows reason in the
  rightmost column. Common: `Resources` (waiting on a free L40S),
  `QOSMaxJobsPerUser` (too many submitted).
- **`uv: command not found` in sbatch script** — venv path wasn't exported.
  The sbatch scripts already do `export PATH="$HOME/.local/bin:$PATH"` but
  double-check it ran.
- **Build errors with torch wheels** — uv resolves the CUDA-enabled wheel by
  default on Linux. If it ends up with a CPU-only torch, add an
  `extra-index-url` for the PyTorch CUDA index in `pyproject.toml`.
