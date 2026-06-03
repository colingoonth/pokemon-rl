# pokerl

A reinforcement learning agent that learns to play Pokemon Red.

PPO on top of [PyBoy](https://github.com/Baekalfen/PyBoy). v1 milestone: earn the first gym badge (Brock).

## Status

Early scaffolding. No training pipeline yet.

## Setup

Requires Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and a vanilla US Pokemon Red ROM.

```sh
uv sync
```

Place the ROM at `roms/pokemon_red.gb`. Verify with:

```sh
shasum roms/pokemon_red.gb
# expected: ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

The ROM is gitignored and not distributed with this repo.

## Layout

```
pokerl/         installable package
  env/          gym env, RAM map, rewards, wrappers
  agent/        networks, PPO trainer
  infra/        config, checkpoint, logging, recording
  eval/         rollout, watch trained agents
  scripts/      train entry point, gif maker
configs/        dev_local.yaml, elsa_brock.yaml
slurm/          ELSA SLURM scripts
tests/          pytest
notes/          design log
roms/           ROMs (gitignored)
```
