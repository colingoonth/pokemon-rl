# Pokemon RL — Primer for new sessions

Read this before touching anything. Then read the linked notes for the
specific area you're working in. Don't duplicate work that's already
captured in those notes.

## The discipline (read first, every session)

**Adding rewards to map direct behavior is wrong, and this project is
way less simple than the V0.3.x approach allowed.** This was the
explicit lesson Colin surfaced 2026-06-05 after five reward versions of
knob-turning failed. When tempted to add a new constant to fix a
specific failure mode, stop and ask whether the right move is
restructuring the reward landscape instead. Curves and structure beat
constants and knobs.

Full reasoning in `notes/design-log.md` under "Day 3 — V0.4". Read that
section before proposing any new reward variant.

## Where the load-bearing context lives

| What | Where | When to read |
|---|---|---|
| Day-by-day decisions + reasoning | `notes/design-log.md` | Always at session start |
| Reward version history table | `notes/reward-versions.md` | When designing a new reward variant |
| Reward class definitions | `pokerl/env/rewards.py` | When implementing a new reward |
| ELSA cluster setup | `notes/elsa-setup.md` | When setting up a new env |
| Vault project MOC | `~/Documents/Documents - Colins Laptop/ObsidianVault/Projects/pokemon-rl/index.md` | When orienting cross-session |
| Recent session logs | `~/Documents/Documents - Colins Laptop/ObsidianVault/Projects/pokemon-rl/` | When picking up where prior session left off |
| V0.4 philosophy decision | `~/Documents/Documents - Colins Laptop/ObsidianVault/Projects/pokemon-rl/pokemon-rl-v04-reward-philosophy.md` | When questioning any V0.4 design choice |

## ELSA cluster realities

- SSH: `ssh guenthc1@elsa.hpc.tcnj.edu` — VPN required off-campus (`vpn.tcnj.edu`)
- Repo lives in `/scratch/guenthc1/pokemon-rl/` on ELSA (NOT home dir — home is small)
- Run outputs go to `/scratch/guenthc1/pokemon-rl/runs/<run_name>/`
- L40S nodes: `gpu-node001-005` (hpe_l40s) + `gpu-node019-021` (l40s). 8 total, 32 cores each, 4 GPUs each.
- **Queue contention varies day-to-day.** Sometimes 4-GPU lands immediately; sometimes it sits in PD(Resources) because long MD-simulation jobs (29-day walltimes) hold partial node allocations across all 8 L40S nodes. *Always check before submitting* — don't assume any specific GPU count will schedule.
- See `notes/design-log.md` Day 2 section on the multi-node DDP attempt for the full multi-node-vs-queue story.
- PyBoy is the actual bottleneck (single-threaded per env). 4×L40S only gets 1.47× over 1×L40S. Don't expect linear GPU scaling.

### Check cluster state before submitting

Before any new submission, run this to see what's free and pick the largest GPU count that will land quickly:

```sh
ssh elsa.hpc.tcnj.edu "for n in gpu-node00{1..5} gpu-node019 gpu-node020 gpu-node021; do echo -n \"\$n: \"; scontrol show node \$n | grep -oE 'AllocTRES=[^ ]+' | head -1; done"
```

Read the `gres/gpu=N` field per node. Free GPUs per node = `4 - N`. Decision rule:

- **Any node with 4 free GPUs (AllocTRES `gres/gpu=0`):** submit the 4-GPU sbatch (`train_brock_*_ddp4.sbatch`). Best throughput.
- **Any node with 3 free GPUs (`gres/gpu=1`):** submit the 3-GPU sbatch (`train_brock_*_ddp3.sbatch`). ~78% throughput of 4-GPU, schedules immediately.
- **Any node with 1-2 free GPUs:** drop to 1-GPU (`train_brock_*_1gpu.sbatch`). ~75% throughput of single 4-GPU job, definitely schedules.
- **Nothing free:** check SLURM start estimate with `squeue -u guenthc1 --start`. If it's more than a few hours out, drop to a smaller GPU count.

The N-GPU variants of every reward config + sbatch are kept in parallel for this reason. Don't pick the GPU count by default — pick it by current availability.

## Key workflows (commands you'll use)

### Check what's running on ELSA
```sh
ssh elsa.hpc.tcnj.edu "squeue -u guenthc1"
```

### Watch a training checkpoint (locally)
**Rule: always use `--speed 0`** (unbounded emulation — see user memory `feedback_pokemon_rl_watch_speed`).

If the run used a non-default start state (e.g. V0.3.0+ uses
`states/blue_fight.state`), pass `--state-path` so the watch starts
from the same state the training did.

```sh
# 1. rsync the checkpoint from ELSA
rsync -avz elsa.hpc.tcnj.edu:/scratch/guenthc1/pokemon-rl/runs/<run_name>/checkpoints/iter_NNNNNN.pt \
    checkpoints/cmp/<descriptive_name>.pt

# 2. watch it — match --state-path to whatever the training config used
uv run python -m pokerl.eval.watch \
    --checkpoint checkpoints/cmp/<descriptive_name>.pt \
    --state-path states/blue_fight.state \
    --steps 5000 --speed 0
```

### Ship a new reward variant
1. Add a new `RewardV0_X_Y` class in `pokerl/env/rewards.py` inheriting
   from the current latest. Set ONLY the constants that change.
2. Register it in `REWARD_REGISTRY` at the bottom of `rewards.py`.
3. Create the config under `configs/elsa_brock_<variant>_<scale>.yaml`
   where `<scale>` is `ddp4`, `ddp3`, or `1gpu` depending on what
   you're going to submit. If you want to keep options open, create
   all three (sed-substitute n_envs/minibatch from the v0.4 baseline).
   Set `reward_class:` to the new class name.
4. Create matching sbatch(es) in `slurm/` by copying the latest working
   sbatch at the target GPU count and updating `--job-name`, the
   RUN_DIR path, and `--config`. Don't decide GPU count in advance —
   check cluster state at submission time and pick the largest
   variant that will schedule fast (see "Check cluster state" above).
5. Verify locally: `uv run pytest tests/ -x` (all 52 tests must pass)
   and `uv run python -c "from pokerl.env.rewards import get_reward_cls; ..."`
   to confirm the class is registered.
6. Commit, push, ssh to ELSA, `git pull`, `sbatch <new>.sbatch`.

### Sync ELSA → local
```sh
./scripts/sync_from_elsa.sh                # all runs
./scripts/sync_from_elsa.sh <run_name>     # one specific run
```

### Cancel a job
```sh
ssh elsa.hpc.tcnj.edu "scancel <jobid>"
```

## Patterns that show up repeatedly

- **`states/` is gitignored** — save state files (`.state`) must rsync separately to ELSA, they don't go through git
- **`--state-path` must be set on both `watch.py` AND in the training config** when using a non-default start state. The eval-script mismatch is a documented bug we already hit (V0.3.0).
- **`async_envs: true` in config + `context="spawn"`** is required for multi-process env collection when CUDA is in use. Don't change this.
- **Per-rank seed offset** lives in `train.py:cfg.seed = cfg.seed + dctx.rank * 1000` — keep it. Without it, all ranks roll identical env trajectories. Bug fix from 2026-06-04.
- **Absolute `state_path` resolution** in `train.py` — `state_path = ROOT / state_path` if relative. Keep this too; relative path was fragile under async subprocess spawning.

## What NOT to do

- Don't propose multi-node DDP without re-reading `notes/design-log.md` Day 2. It's been scoped, scaffolded, and abandoned with reasoning. The expected speedup is modest (~1.7× at 2N) since PyBoy is the bottleneck, and the rendezvous setup is fragile. Re-litigate only if there's a specific reason single-node ceiling actually matters for the experiment.
- Don't hardcode "4-GPU is dead" or "single-GPU is the default." Pick scale by current cluster state at submission time — see "Check cluster state" above.
- Don't bump a single reward constant to fix a behavior. See "The discipline" above.
- Don't run agents (Plan, designer, etc) when a skill covers the task. Most agents have a corresponding skill in `~/.claude/skills/`. Skills are the default.
- Don't run librarian-documentation mid-session. Only at session end, and only via the agent (`/agent-librarian` or the `librarian-documentation` skill), never inline.

## Current project state (as of 2026-06-05 — will go stale)

- V0.4 redesign just shipped. Two jobs running:
  - Job 17439 — V0.4 baseline, `entropy_coef=0.01`, gpu-node019
  - Job 17440 — V0.4 eager, `entropy_coef=0.005`, gpu-node001
- V0.3.x line is the prior approach (knob-turning); design-log Day 3 explains why we pivoted away
- Reward class for the running jobs is `RewardV0_3_4` (in code); `RewardV0_4_0` is an alias going forward
- Curriculum start state: `states/blue_fight.state` (FIGHT menu of Squirtle-vs-Bulbasaur rival battle)
- Live status: `ssh elsa.hpc.tcnj.edu "squeue -u guenthc1"`

This "current state" block will rot. The first thing any session should do is `git log -n 5` and check `squeue` to find out what's actually happening.
