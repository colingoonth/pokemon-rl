# Design log

Day-of decisions and the reasoning behind them, in my own words.
Source for the eventual public writeup.

## 2026-06-03 — Day 1: scaffold, env, PPO, ELSA on

### Why this project

I've wanted to do reinforcement learning for a long time. My ML work so
far has been supervised learning — image classifiers, NLP — and I don't
have anything in RL on my GitHub. RL is a different muscle than
supervised: sparse rewards, long-horizon credit assignment, agents that
learn behavior instead of mapping inputs to labels. Industry ML roles
increasingly want people who can work in both, and "I built an RL agent
that beats Pokemon Red" is the kind of recognizable, hard-but-not-research
project that demonstrates I can ship something difficult.

I picked Pokemon Red specifically because Peter Whidden's well-known
project provides a reference body of work to compare against, the
pret/pokered disassembly maps every RAM address I'd want to read, and
Game Boy emulation is mature (PyBoy runs anywhere). I'm not doing this
for novelty — there's no twist in v1, no new algorithm, no different
game. The differentiator will be the quality of my writeup and the
depth I can explain choices.

### On novelty (and why I'm okay without it)

Pokemon Red RL is the canonical RL portfolio project — Peter Whidden's
well-known implementation has been seen by millions, and dozens of
derivatives exist on GitHub. I'm doing it anyway, deliberately, for
industry ML targeting. The reasoning:

1. **My career goal is industry ML, not ML research.** For industry
   roles, hiring grades on whether you can ship a complete hard project
   end-to-end, defend every choice in technical depth, work with real
   infrastructure (HPC, multi-process training, SLURM), write code a
   senior engineer would respect, and debug hard performance issues.
   None of those are novelty signals — they're depth and execution
   signals. A strong execution of a known problem beats a weak
   execution of a novel one.

2. **Whidden's project being famous is a feature, not a bug.** Hiring
   managers immediately know the difficulty class without me having to
   explain it. The reference exists for direct comparison. "I
   implemented the canonical problem with X engineering rigor and Y
   empirical results" is concretely evaluable.

3. **This is my "I have RL experience" piece, not my "I do novel
   research" piece.** Future personal projects can be the novelty piece
   (Pokemon Crystal RL with a real differentiation, or PPO vs DreamerV3
   comparison, or RL applied to a non-game domain). Together those
   become a stronger story than either alone — depth shown here,
   originality shown there.

What would make this a weak portfolio piece: shallow writeup,
AI-generated boilerplate code, no eval methodology, inability to defend
hyperparameters. What makes it strong: thoughtful writeup, clean
test-covered code, honest empirical results, deep technical defense of
every choice. I'm targeting strong.

### Scope

v1 milestone is the **first gym badge (Brock)**. I picked this over
"beat the Elite Four" deliberately because (a) I have other personal
projects I want to ship this year, (b) Brock is a recognizable
milestone that anyone who's played Pokemon understands, and (c)
shipping a complete, polished project beats half-finishing a larger
one. If V1 ships well and I want to keep going, the infrastructure
supports it — badge-by-badge updates with new training runs and
writeup additions. But the scoped commitment is Brock.

### Game choice

I went with vanilla Pokemon Red (US, SHA1
`ea9bcae617fdf159b045185467ae58b2e4a48b9a`) over the Colorization hack,
the Enhanced hack, or FireRed. The reasons are pragmatic: any
modification to the ROM shifts RAM addresses unpredictably, breaking
the published memory maps; FireRed is Game Boy Advance which PyBoy
doesn't emulate at all; and all the prior art (Whidden's writeup,
pret/pokered disassembly, community Pokemon RL repos) targets vanilla
Red. Diverging from vanilla means losing all of that reference
material in exchange for cosmetic differences the agent doesn't care
about.

### Algorithm choice

PPO. The reasons: it's the well-documented on-policy default for
environments like this (discrete actions, image observations, sparse
rewards), CleanRL's `ppo_atari.py` is a clean single-file reference I
can derive my own implementation from (clean-room rule — I read it for
the approach but type out my own code), and it's the same family of
algorithm Whidden used so I can compare. I'm explicitly not picking
something exotic like DreamerV3 or MuZero for v1. Algorithm-novelty
isn't the project. Understanding PPO deeply enough to defend it is.

### Reward function v1

When I first thought about reward signals before knowing anything
formal about RL, my gut said "catching pokemon or beating trainers...
and maybe discovering new areas." That instinct turned out to map
almost exactly onto how Whidden and similar projects design their
reward functions: exploration + combat progress + story milestones.
RewardV0_1 codifies those categories with these starting weights:

| Signal | Weight | Reasoning |
|---|---|---|
| New `(map_id, x, y)` tile visited | +1 | Dominant signal early-game — most progress is "go somewhere new" |
| Total level gained across party | +5/level | Encourages combat, but kept small enough that grinding Pidgey forever isn't optimal |
| New event flag set | +10/flag | Story progress — getting the starter, beating rival, etc. |
| Badge earned | +100 | The jackpot — explicit milestone signal |
| Per-step penalty | -0.001 | Don't dawdle. Small enough not to dominate but present |

These weights are an educated starting point, not a defended answer.
I expect V1 to fail in specific ways — the agent will likely find one
signal it can cheese (probably exploration, where it walks back and
forth at a map boundary) and the others won't matter. The interesting
ML work in this project is iterating to V2, V3, etc. based on what
training reveals. I'm deliberately starting with the simplest version
that compiles so I have something to compare against.

Baselines are captured on `reset()` so the agent isn't rewarded for
state already present in the save file (post-intro starter, $3175,
level 6).

### PPO hyperparameters

Starting values come from CleanRL's `ppo_atari.py` (the canonical
single-file PPO reference for image-observation environments), with one
deliberate change for Pokemon Red:

| Hyperparameter | Value | Source / reasoning |
|---|---|---|
| `gamma` (discount) | 0.999 | **Raised from CleanRL's 0.99.** Pokemon Red is long-horizon — beating Brock takes thousands of steps from spawn. Higher gamma makes future rewards count more, which matters when the badge reward is many steps away. |
| `gae_lambda` | 0.95 | CleanRL default. Standard PPO advantage-variance tradeoff. |
| `clip_coef` | 0.1 | CleanRL default. Tighter clipping = more conservative policy updates. |
| `entropy_coef` | 0.01 | CleanRL default. Encourages exploration in the action distribution. |
| `value_coef` | 0.5 | CleanRL default. Standard relative weight on value loss. |
| `learning_rate` | 2.5e-4 with linear anneal | CleanRL default. Anneals to 0 over the run. |
| `n_steps` (rollout length) | 256 (ELSA) / 128 (dev) | Bigger n_steps on ELSA gives more on-policy data per update. |
| `n_envs` | 32 (ELSA) / 4 (dev) | ELSA L40S can handle 32 parallel async envs. |
| `n_epochs` | 4 | CleanRL default. |
| `minibatch_size` | 512 (ELSA) / 64 (dev) | Scales with `n_envs * n_steps`. |
| `max_grad_norm` | 0.5 | CleanRL default. Gradient clip for stability. |

The honest defense for these values: PPO is a well-tuned algorithm and
most defaults work across a wide range of environments. The CleanRL
implementation details paper documents this. The one knob I touched
(gamma) is the one most-clearly mismatched between Atari (short
episodes) and Pokemon (long episodes). I expect to tune more values
once training reveals what's bottlenecking learning — but starting
from a known-good baseline beats starting from a random point.

### Network architecture

Standard NatureCNN backbone (from the original DQN-on-Atari paper)
feeding two heads:

```
Input:  (4, 72, 80) uint8      — last 4 grayscale frames stacked as channels
  -> Conv2d(32, 8x8, stride 4) + ReLU
  -> Conv2d(64, 4x4, stride 2) + ReLU
  -> Conv2d(64, 3x3, stride 1) + ReLU
  -> Flatten
  -> Linear(_, 512) + ReLU
  -> branches:
     - actor head:  Linear(512, 7) -> action logits
     - critic head: Linear(512, 1) -> V(s)
```

Frame stacking (4 frames) is what lets the network perceive motion — a
single Game Boy frame is a static scene. The downsample from 144x160 to
72x80 keeps the convolutional input manageable. Observations are uint8
in the buffer; the network divides by 255 inside `forward()` to scale
to `[0, 1]`.

Initialization follows the well-documented PPO implementation details:
orthogonal weights with gain `sqrt(2)` on hidden layers, gain `0.01` on
the actor output (so the initial policy is near-uniform — entropy =
log(7) ≈ 1.946 — instead of being arbitrarily biased), gain `1.0` on
the critic output. These small details are the kind of thing that
takes PPO from "kind of works" to "trains reliably," and I picked them
up from CleanRL's implementation.

I am NOT using RAM-extracted features as an auxiliary observation in
v1. The architecture above only sees pixels. Adding a structured-feature
head later (party levels, badges, position as a feature vector
concatenated before the actor/critic heads) is a clean v1.5 experiment.

### Compute split

- **Macbook for development**: 4 parallel envs, headless PyBoy, fast
  iteration. The dev loop is "tweak reward / hyperparam → run 30-second
  smoke → look at output → repeat." Watching trained agents in the
  SDL2 window happens here too.
- **ELSA L40S for training**: 32 parallel envs (async,
  subprocess-per-env), single L40S GPU, multi-hour runs.

ELSA setup (one-time): SSH key registered with GitHub, `uv` installed
at `~/.local/bin/uv`, repo cloned to `/scratch/guenthc1/pokemon-rl`
(ELSA's home dir is too small for ML projects), ROM + post-intro save
state pushed via `scp`. SLURM jobs request
`--partition=gpu --gres=gpu:1 --constraint=l40s` to land on the right
hardware.

### Why async vector envs (the debugging story)

First real training run on ELSA used `gym.vector.SyncVectorEnv` (envs
step sequentially in one Python process). Even with GPU utilization
showing 96%, throughput was ~32 sec/iter for 8192 timesteps —
projecting to **~9 days** to finish the 200M-timestep budget, well past
the 48h SLURM limit. Root cause: the Python GIL serializes 32 PyBoy
steps into one process, so we get effectively single-env throughput.

Fix: `gym.vector.AsyncVectorEnv` (each env in its own subprocess via
multiprocessing). Same code path otherwise. Result: **~3 sec/iter, ~10x
speedup**, projecting to ~20 hours for the full budget — comfortably
within the SLURM time limit.

Lesson learned: when scaling RL with CPU-bound emulators, GIL is the
dominant bottleneck before GPU is. Sync vector envs are fine for unit
tests and small smoke runs; anything past 4-8 envs should be Async.

### Open questions / what's next

Training run 13447 is running on ELSA as of 2026-06-03 evening,
projected to finish ~20 hours of wall-clock time. What I expect to see
in tomorrow's metrics, and what each outcome implies:

- **Best case**: entropy drops meaningfully (1.94 → 1.5ish), unique
  tiles climb past spawn count, mean return turns positive once
  exploration reward outpaces step penalty. This would confirm V1's
  reward signal is at least *trainable*, and we move to longer /
  better-tuned runs.
- **Most likely case**: entropy slowly drops but tiles barely move
  because the agent learns one degenerate strategy (probably
  button-mashing at spawn). This is the expected V1 failure mode and
  motivates RewardV0_2.
- **Worst case**: training diverges (NaN losses, entropy collapses to
  0, episodes complete instantly because the agent picks 'A' forever
  in a menu). Would imply a real bug, not a reward-design problem.

#### RewardV0_2 hypotheses (to design once we see V1's actual failure mode):

1. **Diminishing exploration reward**: instead of +1 per new tile flat,
   decay the reward for visiting tiles near already-visited regions.
   Forces meaningful movement, not just edge-walking.
2. **Movement-required reward**: +1 only when position changes between
   consecutive steps. Penalizes menu-staring.
3. **Curriculum-style milestone bonuses**: hand-defined bonuses for
   "left Pallet Town," "entered Viridian," "entered Mt. Moon," etc.
   Closes the gap between sparse natural rewards.
4. **Negative reward on fainting**: -10 when party HP hits 0 (matters
   once combat is happening).

#### Other open questions:

- Should we randomize starter Pokemon at reset to prevent over-fit to
  one seed?
- Should we add a domain-randomized intro state (different positions,
  different party levels) for v1.5?
- At what badge milestone (if any) does the project transition from
  "PPO + tuning" to "fundamentally different algorithm needed"?
