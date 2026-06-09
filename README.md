# pokerl

A reinforcement learning agent that learns to play Pokemon Red.

PPO on top of [PyBoy](https://github.com/Baekalfen/PyBoy). Milestone: earn the first gym badge (Brock).

This is the **`vcl` branch** — set up for single-GPU runs on NC State's [Virtual Computing Lab](https://vcl.ncsu.edu/) (RTX 2080 Ti, 16 cores). For ELSA/SLURM, switch to the `elsa` branch.

---

## First-time setup on a new VCL reservation

Reserve an **Ubuntu 22 GPU with CUDA (GeForce RTX 2080 Ti)** node, then SSH in:

```sh
ssh cguenth2@<ip-from-vcl-portal>
```

Clone the repo and run the setup script:

```sh
git clone <repo-url> && cd pokemon-rl
git checkout vcl
./scripts/setup_vcl.sh
```

`setup_vcl.sh` installs tmux, uv, and all Python deps. Run it once per reservation.

Place the ROM at `roms/pokemon_red.gb` (gitignored, not in the repo). Verify:

```sh
shasum roms/pokemon_red.gb
# expected: ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

---

## Running a training job

```sh
./scripts/launch_vcl.sh configs/vcl_brock_v0_4_2_center_1gpu.yaml
```

This launches training in a detached tmux session so it keeps running if you disconnect. To check on it:

```sh
tmux attach -t train                                                    # attach to live output
tail -f runs/vcl_brock_v0_4_2_center_1gpu/train.log                    # or just tail the log
```

---

## Syncing results back locally

```sh
./scripts/sync_from_vcl.sh cguenth2@<ip>                               # all runs
./scripts/sync_from_vcl.sh cguenth2@<ip> vcl_brock_v0_4_2_center_1gpu  # one run
```

---

## Watching a checkpoint

Pull a checkpoint locally, then:

```sh
uv run python -m pokerl.eval.watch \
    --checkpoint checkpoints/<name>.pt \
    --state-path states/blue_fight.state \
    --steps 5000 --speed 0
```

`--speed 0` is required (unbounded emulation speed).

---

## Configs

| Config | Description |
|---|---|
| `vcl_brock_v0_4_2_center_1gpu.yaml` | V0.4.2 center arm — primary hypothesis, diagnostic probe (entropy 0.005) |
| `dev_local.yaml` | Quick local smoke test, no GPU required |

Add new VCL configs with the `vcl_` prefix and `n_envs: 12` (all 16 cores available, no GPU sharing).

---

## Layout

```
pokerl/         installable package
  env/          gym env, RAM map, rewards, wrappers
  agent/        networks, PPO trainer
  infra/        config, checkpoint, logging
  eval/         rollout, watch trained agents
  scripts/      train entry point
configs/        vcl_*.yaml, dev_local.yaml
scripts/        setup_vcl.sh, launch_vcl.sh, sync_from_vcl.sh
tests/          pytest
notes/          design log, reward version history
roms/           ROMs (gitignored)
states/         PyBoy save states (gitignored)
```

---

## Branch strategy

| Branch | Cluster | Notes |
|---|---|---|
| `vcl` | NC State VCL (RTX 2080 Ti, single GPU) | This branch |
| `elsa` | TCNJ ELSA (L40S, SLURM, multi-GPU) | slurm/ scripts, DDP configs |
| `main` | — | Shared code; merge reward changes here first |
