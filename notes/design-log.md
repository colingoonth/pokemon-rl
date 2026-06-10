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

## 2026-06-04 — Day 2: V0.3.x curriculum + abandoned multi-node DDP

### The reward arc up to today

V0.2.7 (BEAT_MON +20, TRAINER_WIN +30, FLEE_PENALTY -1) was supposed
to break V0.2.6's flee-everything behavior by dropping the fight
breakeven from 74% to 60% win confidence. After 3500 iters / 29M
steps it was still flee-dominant. The math was right but the
bootstrap was wrong: agent needed to fight to learn win-prob > 60%,
but wouldn't fight until it believed it could.

### V0.3.0: curriculum fork

The cleanest break from V0.2.x: change the start state. Captured a
save state at the FIGHT menu of the Squirtle-vs-Bulbasaur rival
battle (`states/blue_fight.state`). Rival battle is unavoidable, so
combat reward is the only signal early on. Plumbed `state_path`
through `PPOConfig -> train -> make_vec_env` (one new field on the
config dataclass).

I labelled this V0.3 instead of V0.2.8 because changing the start
state changes the *task*, not the reward function. V0.2.x was
"tune the reward." V0.3.x is "change the curriculum." Different
axis, different version namespace, cleaner writeup later.

V0.3.0 results: reached V0.2.7's ceiling ~3.5x faster (iter 1000 /
8M steps to hit +450 vs V0.2.7's iter 3500 / 29M to hit +440), but
return then oscillated 260-530 instead of climbing. Watching a
mid-training checkpoint showed the agent correctly going into fights
and fleeing only when they looked unwinnable. That's *rational* EV
behavior, not a failure — it's the V0.2.7 reward function working as
designed at the 60% breakeven. The issue is the threshold is still
too high for early-stage Squirtle against most of Route 1.

### V0.3.1: FLEE_PENALTY -1 -> -5

Two reward-shaping options to push past V0.3.0's plateau:

1. **DAMAGE_DEALT shaping** — small per-attack reward. Rejected:
   would teach "Tail Whip = 0 damage = bad" and close off the
   status-move branch (Tail Whip stacking is genuinely optimal in
   some L5 matchups; can't bake in a "damage = good" prior).
2. **FLEE_PENALTY bump** — only penalizes the leave-battle action,
   doesn't bias in-battle action selection. Status-move learning
   preserved. Drops fight-vs-flee breakeven from 60% to ~45%.

Picked (2). Launched as RewardV0_3_1 with FLEE_PENALTY = -5.

### Bug found mid-flight: eval-time start-state mismatch

When I tried to watch the V0.3.0 policy, it looked like the agent
was starting in Pallet Town, not the FIGHT menu. Almost concluded
training was broken too. Actually `watch.py` had never been wired
to accept `--state-path` — it used the env default
(`post_intro.state`). Training was using `blue_fight.state`
correctly; only eval was mismatched. Added the `--state-path` flag
and confirmed first frame is Squirtle vs Bulbasaur as intended.

Lesson for V0.4 onward: any time we add a config knob that affects
the training environment, the eval scripts need the same knob.
There's no shared "env config" abstraction — make.py takes raw
kwargs at each call site.

### Multi-node DDP: tried and abandoned

After single-node 4xL40S hit its 1.47x ceiling (PyBoy CPU bound),
multi-node was the obvious next lever. ELSA has 8 L40S nodes total
(gpu-node001-005, 019-021), each 32-core 4-GPU. Theoretical 8x
ceiling.

Scaffolding built before launch:
- `pokerl/scripts/bench_nccl.py` — 30-line rendezvous + all_reduce
  bandwidth probe
- `slurm/sanity_multinode.sbatch` — 2-node, 10-min, no-training
  sanity test
- `slurm/train_brock_bench_2n.sbatch` + `configs/elsa_brock_bench_2n.yaml` —
  2N throughput bench at 48 envs / mb=768 / lr=3.06e-4 (sqrt(1.5)
  scale per Hilton 2021), 10M timestep budget

Three reviews flagged issues:
- **SLURM/torchrun:** sbatch pattern was mostly right but needed
  `--rdzv-id=$SLURM_JOB_ID`, `NCCL_IB_DISABLE=1`,
  `NCCL_SOCKET_IFNAME=^lo,docker`, per-job MASTER_PORT, and
  `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600`. Applied all.
- **PPO at scale:** "minibatch ∝ n_envs" alone is insufficient — PPO
  is not batch-size invariant. Use sqrt-LR scaling from launch (not
  wait-and-see), bump entropy_coef proportionally to counter
  premature sharpening, log clip_fraction/approx_kl/explained_variance
  as scaling tripwires.
- **End-to-end risk:** found two real bugs in train.py — all ranks
  using identical seed (wasted parallelism) and relative `state_path`
  being fragile under async-subprocess cwd inheritance on multi-node.
  Both fixed; kept the fixes because they're correctness fixes that
  apply single-node too.

**Why I abandoned:**

Two independent reviews projected realistic 2N speedup at ~1.7x and
8N possibly *slower* than 4N due to NCCL Ethernet overhead. So the
upper bound on multi-node speedup was modest to begin with.

Then queue contention killed the math. Submitted the 2N sanity test
(needs 2 *whole* L40S nodes simultaneously). All 8 L40S nodes were
in `mix` state (partially allocated) with a backlog of dozens of
queued jobs from other users — sanity sat in PD with reason
`(Priority)` for ~15 min before I bailed. Even a 1.7x throughput
gain doesn't pay for a queue wait of 30+ min per launch when single-
node submits run within seconds. The "iterate faster" goal gets
eaten by queue time.

Scaffolding deleted on this date. Reviews + decision preserved here
so I don't reconsider this if I forget the cluster contention story.

**What I kept from this thread:**

1. The two train.py bug fixes — per-rank seed offset
   (`cfg.seed + dctx.rank * 1000`) and absolute `state_path`
   resolution against `ROOT`. Real bugs, ship them.
2. The PPO-at-scale notes — sqrt-LR, entropy bump, tripwire metrics.
   Won't need them now but worth remembering if I ever do
   reconsider multi-node.
3. The cluster topology map (8 L40S nodes, all 32-core, no
   Infiniband). Documents reality for future me.

**What would change my mind on multi-node:**

- ELSA gets less contested (semester ends, summer break, etc.)
- Single-node ceases to be a bottleneck (e.g., we pivot off PyBoy
  to a parallelizable emulator, removing the 1.47x cap)
- A multi-day training run becomes the experiment unit (writeup
  finale, demo for portfolio) where the queue wait is a one-time
  cost amortized over many GPU-hours

## 2026-06-05 — Day 3: V0.3.2/3 sweeps, hit the wall, full redesign to V0.4

### Where V0.3.x got us

The V0.3.x line (curriculum start at Blue battle + reward-magnitude
tuning) was supposed to make the agent fight and heal. By V0.3.3 it
fought well, didn't heal, and didn't progress past Route 1.

Three quick observations from watching V0.3.3 at iter 10800 (~66M
steps in):

- **It will go into fights even at low HP**, then realize partway
  through it's losing, and try to run. The flee-penalty bump worked.
- **It does not head for a PokeCenter when its HP is low.** Even
  with HEAL_QUAD_COEF=12 (Whidden-shaped quadratic HP-restored
  reward), the agent never *finds* the heal chain. Dying is fine
  because kills pay enough that fainting nets out positive over
  the course of an episode.
- **It doesn't make it to the next town.** Wanders Route 1 in
  loops, fights, dies, respawns, repeats.

### The wall I hit, in EV terms

The reason V0.3.3 stalled is the per-cycle EV math:

```
Per-kill reward:         +20
Per-faint+blackout cost: -25 + -7 = -32
Typical episode:         5-10 kills before fainting
Per-cycle net EV:        +100 to +200 (positive)
```

Dying is *profitable*. The agent is correctly maximizing return by
fighting until it dies. No amount of healing-reward bumping fixes
this because the agent never *experiences* the heal chain — random
exploration from Route 1 to a PC + nurse-dialog is a low-probability
action sequence, and the +12 quadratic-heal payoff is dwarfed by
the +20 per-kill stream right next to it.

### The framing shift to V0.4

I stepped away from the project for a while and came back to it
with a clean head. Roughly two hours of thinking from a full
restart — not iterating on the V0.3.x design but deciding what I
actually believed the reward function should look like. I
already knew I wanted a complete rework before I sat down. The
choices below are mine; I deliberately wasn't trying to crib
Whidden, and most of what I landed on doesn't match his published
shape anyway. (Where it does overlap — quadratic heal, softer
death penalty — those were lookups I made *after* deciding on
the structure, mostly to sanity-check that someone else had
arrived at similar conclusions.)

The core thesis I came back with: **direct per-action rewards
train the agent to pick a confirmed ideal route towards a win.**
That's the V0.3.x failure in one sentence. Each knob-turn —
BEAT_MON +20, FLEE_PENALTY -5, FAINT_PENALTY -25 — taught the
policy to optimize a specific behavior the reward designer
already knew about. By the time I'd shipped V0.3.3 I was, in
effect, hand-coding "fight, then flee if losing, then go heal" —
the agent never had to *discover* any of it. And every time my
hand-coded policy hit a corner case (no PC visits, no story
progress) my response was to bolt on more reward shaping for
the next failure.

I want the agent to seek future value through chains of state, not
follow per-step instructions I built into the reward. That's the
philosophical pivot. The mechanical translation is:

I spent about an hour of that two-hour window in Desmos. This
wasn't decorative — it was the load-bearing part of the design.
The reason every prior V0.x reward function felt arbitrary is
that I was picking constants by gut ("does +20 sound right for a
kill?") and then arguing about them. Desmos let me actually plot
the curves and ask the real questions:

- **What does each reward curve look like over its useful
  domain?** For levels: log_6(level) is gentle and slow. At L5
  it's 0.9; at L15 (Brock-ready) it's 1.51. The increment per
  level is small (~0.1 around L5-15) — that's what I want: the
  agent shouldn't be obsessed with leveling, just nudged toward
  it.
- **How do the cumulative magnitudes compare?** This is where
  Desmos earned its keep. I plotted integrals/sums for each
  reward over realistic game state. Cumulative BADGE for all 8
  badges sums to ~450 (15 × x^1.3 from x=1 to 8). Cumulative
  CATCH for the full pokedex sums to ~663. Cumulative EXPLORE
  depends on N — at N=1000 tiles it's ~250, at N=3000 it's
  ~2250, at N=5000 it's ~6250. **This is how I caught that
  exploration was dominating everything by an order of
  magnitude** before launching: my first coefficient was 0.0005
  and the Desmos plot made it obvious that at any realistic
  game-completion tile count, exploration would drown out badges
  by 5-10×. Dropped to 0.0001 and the totals lined up.
- **What's the relative payoff at each decision point?** For the
  per-area kill cap, I plotted the falling log over kill 1-5 and
  asked "what's the marginal value of the 5th kill versus
  walking out of the area?" At kill 5 it's 0.76. Walking is
  worth one new-tile reward (~0.5 mid-game). The two are
  intentionally close — that's the engineered indifference point
  where the agent should pick whichever serves the longer chain.
- **Where do reward components compete?** The falling per-area
  kill curve was a direct result of seeing on Desmos that a
  rising curve would push the agent to stay-and-grind (the V0.3.x
  failure mode), while a falling curve makes the *first* fight
  in any new area worthwhile but caps the value of camping. I
  considered both directions in the graph before picking falling.
- **Are the discounted future values reachable?** PPO with
  γ=0.999 propagates ~256-step horizons cleanly. Anything past
  that gets heavily haircut. The Desmos work involved checking
  that the curves' relevant payoffs land *within* that horizon
  for a reasonable policy. Healing chain (~50 steps), per-area
  kill cap (~30 steps), badge after Brock (~500+ steps,
  partially reachable) — I picked formulas where the credit
  assignment chain was achievable.

The Desmos sheets are throwaway but the process wasn't. Every
formula in V0.4 has a curve I looked at and a magnitude I checked
against the others. Constants weren't guessed; they came out of
the comparison. That's the thing the V0.3.x line was missing.

### The bigger lesson

Beyond the specific design pivot, the V0.3.x → V0.4 transition
forced a meta-realization I should keep in front of me for the
rest of this project:

**Adding rewards to map direct behavior is wrong, and this
project is way less simple than my approach allowed.**

I was treating Pokemon Red RL as a hyperparameter sweep. Pick a
behavior I want, write a reward for it, tune the magnitude until
the agent does the thing. Five reward versions in (V0.2.2 →
V0.2.7 → V0.3.0 → V0.3.1 → V0.3.2 → V0.3.3), all that approach
produced was a stack of behavior-shaping knobs that each fixed
one corner case and created the next. Bump BEAT_MON to teach
fighting → agent never flees. Bump FLEE_PENALTY to fix that →
agent fights until it dies. Bump healing reward to fix *that* →
agent never finds the heal chain because it's too far away to
discover.

The pattern: every "fix" was me encoding more of the game's
strategy into the reward function. The agent never got the
chance to discover anything; it was being told.

The actual problem isn't "what's the right magnitude for
BEAT_MON" — it's "what reward landscape allows the agent to
discover correct behavior on its own." That's a fundamentally
harder design problem than the one I'd been solving for two
days. It needs curves, not constants. It needs structure, not
knobs. It needs me to leave room for the agent to learn the
game, not to encode the game into the rewards.

That's the thing I'd been resisting, probably because the
knob-turning route was tactile and produced visible iteration
per day. Stepping back and admitting "this is harder than I was
treating it" cost me an evening of training-time and gave me
nothing immediately to ship. But it's the only way forward that
doesn't end with a sixteen-knob reward function that ships a
brittle, hand-coded policy.

The discipline I want to hold for the rest of this project:
**when I notice myself adding a new reward constant to fix a
specific failure mode, stop and ask whether the right move is
restructuring the reward landscape instead.** Knob-turning was
the wrong tool for V0.3. It will be the wrong tool again.

### The philosophy in V0.4 design:
1. **Remove dense per-action rewards** like BEAT_MON +20. Replace
   with a capped, falling log over per-area kills so the agent
   can't grind a single route. Combat becomes instrumental, not
   the dominant signal.
2. **Reward state in curves, not constants.** log_6(level) for
   leveling, log_2.5(pokedex) for catches, 0.0001 × N^1.001 per
   tile for exploration (growing with total tiles seen), 15 × x^1.3
   per badge. Each one rewards *what you've accumulated*, not
   *what you just did*.
3. **Force future-seeking through chain-of-value.** The agent has
   to fight to gain XP, gain XP to level, level to win, win to
   reach badges. Each link individually pays small; the cumulative
   chain only adds up if you actually progress.
4. **Strip everything that doesn't carry weight.** Remove FLAG,
   NEW_MAP, MOVE_BONUS, STEP_PENALTY, MART_PC_BONUS, PC_FIRST_VISIT,
   GYM_BONUS, NEW_CATCH_BONUS, NEW_ENCOUNTER. One-shot rewards
   don't drive recurring behavior; tiny rewards just dilute the
   gradient signal. Keep the reward landscape lean.
5. **Soften the death penalty.** Whidden's documented finding was
   that big death penalties create the "agent fears the PC because
   Pokemon got deposited there during a forced blackout heal"
   trap. So FAINT -25 -> -5, LOSE -7 -> -20 (blackout itself is
   bad but individual faints are non-fatal lessons).

### V0.4 reward landscape

| Category | Formula | Notes |
|---|---|---|
| HEAL_QUAD | `(delta_hp_frac)^2 × 15` | Whidden-style continuous, fires on any HP rise |
| LEVEL | `log_6(new_level)` per unit | At L6: 1.0. At L15: 1.51. ~13 cumulative L5→15. |
| CATCH | `log_2.5(new_pokedex_count)` | First new catch (pokedex=2): 0.76. All 151: ~663. |
| EXPLORE | `0.0001 × N^1.001` per new tile | At N=1000: 0.50/tile. At N=3000: 1.52. |
| BEAT_MON | `log_2.5(7 - kill_n)`, cap 5/area | Falling. Kill 1: 1.96. Kill 5: 0.76. Total per area ~7.2. Per `map_id`. |
| BADGE | `15 × x^1.3` after gaining | 1st: 15. 2nd: 37. 8th: 224. Superlinear. |
| POKEBALL_BUY | `5 × max(0, 1 - n/20)` | Per ball, n=owned before. Linear decay, 0 past 20. |
| STUCK | `-0.005/step` after 200 steps with no new tile | Activity-based, catches tight wander loops. |
| TRAINER_WIN_BONUS | `+10` (was +30) | Sparse but solid signal. |
| FLEE_PENALTY | `-0.5` (was -5) | Strategic retreat is cheap now. |
| FAINT_PENALTY | `-5` (was -25) | Whidden softening. |
| LOSE_BATTLE | `-20` (was -7) | Blackout still costly. |

Disabled (set to 0): NEW_MAP_REWARD, MOVE_BONUS, STEP_PENALTY,
MART_PC_BONUS, PC_FIRST_VISIT, GYM_BONUS, FLAG_REWARD,
NEW_CATCH_BONUS, NEW_ENCOUNTER, BEAT_MON_REWARD (replaced by
capped log), LEVEL_REWARD (replaced by log_6), CATCH_REWARD
(replaced by log_2.5), EXPLORE_REWARD (replaced by power-law),
BADGE_REWARD (replaced by power-law).

### Key design decisions and the why

**Why the falling per-area kill curve (rising would be the obvious
choice for "clear the route"):** I considered both directions.
Rising would reward completion — finish the area, get the big
payoff. Falling rewards the *engagement decision* — first kill in
a new area pays most, and there's diminishing return on staying
to grind. I chose falling because the failure I'm fixing is
"agent grinds Route 1 forever"; the curve has to discourage
extended camping, and falling does that more cleanly.

**Why 0.0001 not 0.0005 on the explore coefficient:** I worked
with 0.0005 in Desmos, but Claude flagged that at N=3000 tiles
(realistic mid-training), cumulative exploration reward would
be ~2250 — 5x larger than total BADGE cumulative across all 8
badges. Exploration would dominate everything. Dropping to 0.0001
puts cumulative exploration at ~450 at N=3000, comparable in
magnitude to badge totals. Better-balanced reward landscape.

**Why I'm keeping a small FAINT_PENALTY at all (-5):** Whidden
removed it entirely. The argument for keeping a small one: the
quadratic heal reward + blackout cost are downstream signals; a
direct, small penalty on faint helps the value function attribute
the cost to the *decision* that led to the faint, not just the
terminal blackout. -5 is small enough that the V0.3.x failure
mode (death is profitable) doesn't recur, but big enough to
register on the per-step gradient.

**Why I'm not adopting Whidden wholesale:** his shipping version
disables BEAT_MON_REWARD entirely. I keep a capped version because:
(a) some early-game gradient on combat is needed before the
log_6(level) signal kicks in; (b) without any per-kill payoff the
agent might never learn to fight at all in the rival battle (the
curriculum start state). The cap at 5/area is the compromise —
combat learning happens, but grinding stops being profitable.

**Why entropy_coef 0.005 + 0.01 sweep:** the redesign is sparser
than V0.3.x. Lower entropy should help the policy commit to the
discovered value chains faster, but too low and it locks in
before discovery. Running both 0.005 (eager) and 0.01 (baseline)
in parallel to see which entropy level matches the new reward
density.

### What "working" looks like for V0.4

V0.4 won't be evaluated by mean_return the way V0.3.x was — the
units changed. Expecting V0.3.3's +750 plateau here is wrong;
V0.4 might cap at +100-200 even when working, because the
reward magnitudes are smaller.

Real signals to watch:
1. **Tile count climbs past V0.3.x's per-iter ceiling (~25K
   all-envs).** If V0.4 explores more, the exploration curve is
   working.
2. **Episode length increases.** Agent staying alive longer means
   the future-seeking design is paying off.
3. **First badge event** — visible in train.out as a +15 reward
   jump in any single env. V0.3.x never got there.
4. **Pokedex count increasing in watch.py** — proves catch
   reward fired and the agent did something with it.

Patience window: V0.4 will take longer to read than V0.3.x. Credit
assignment over 100+ step chains takes many episodes. Honest
estimate: 6-8 hours for first read, 24+ hours for confident
verdict. The reward signal is real but sparse.

### What I'm explicitly betting on

That removing dense per-kill rewards forces the agent to find
the longer chain (kill → XP → level → progress → badge), and
that the curve-shaped exploration reward pulls it past Route 1
naturally without me having to hand-design "go north to Viridian."

The bet might fail in two ways:
- Agent never learns to fight at all (too sparse), policy
  collapses to a flee-everything degenerate strategy.
- Agent finds a degenerate exploration exploit (some grid pattern
  that registers as "new tiles" via emulator quirk) and farms it
  indefinitely.

Both are recoverable. The bet is worth making because dense action
shaping has now failed three times in a row (V0.2.7 → V0.3.0 →
V0.3.1 → V0.3.2 → V0.3.3), and the project deserves an honest
philosophical pivot rather than a sixth knob-turn.

### Code housekeeping

Class is named `RewardV0_3_4` in code (since it was built before
the V0.4 rename). Kept as the actual class for the running jobs;
added `RewardV0_4_0 = RewardV0_3_4` alias going forward so new
configs use the conceptually-correct version label.

## 2026-06-05 — Day 4: V0.4 Tail Whip diagnosis + V0.4.1 fix

### Where the day started

Yesterday's V0.4 launched two parallel jobs: 17439 (baseline,
entropy 0.01) and 17440 (eager, entropy 0.005). By morning 17439
was dead — PyBoy `PyBoyAssertException: No data` from `BaseMBC.load_ram`
at startup. Then NCCL timed out waiting for rank 0 and SIGTERM
cascaded across the DDP group.

17440 was still running fine — won its race, kept going.

### The PyBoy NFS race

Root cause: PyBoy auto-creates a `<rom>.gb.ram` companion file to
persist save-RAM. With 24 PyBoy subprocesses per rank × 2 jobs all
racing for the same `/scratch/.../roms/pokemon_red.gb.ram` over NFS
at startup, one of them got a partial read and crashed. 17440 won
its race by luck; 17439 lost.

Fix: pass `ram_file=BytesIO(b"\x00" * 32768)` (Pokemon Red MBC3 has
32KB battery-backed save RAM). Each env subprocess gets its own
in-memory buffer, no disk contention at all. Note: empty `BytesIO()`
does NOT work — PyBoy's `load_ram` reads the cart-RAM size during
init and raises "No data" on short read. Pre-sized zero buffer is
required.

Resubmitted as 17653. Boots clean, race eliminated.

### Watching V0.4 eager at iter 5000

17440 (eager, entropy 0.005) hit iter 5000 at +158 mean_return with
H ~0.95. Watched a checkpoint via watch.py with `--state-path
states/blue_fight.state`.

Result: **agent is spamming Tail Whip until it dies.**

Specifically:
- Against Blue rival: won by Tackling
- After Blue, almost every wild battle: Tail Whip until faint
- One exception: a single wild battle won with Tackle by chance
- Behavior outside battle: heads straight to the grass on Route 1

So the overworld policy is *working* — agent goes seeking
encounters, exploring the curve-shaped EXPLORE reward. The
in-battle policy is broken — Tail Whip-locked.

### How does it earn +158 if it's Tail Whip spamming?

This was the question that cracked the diagnosis open. The +158
isn't from combat — it's from post-Blue exploration. The Blue
rival fight gives TRAINER_WIN_BONUS +10 + falling-log first-kill
~+2. After winning, the agent walks Pallet → Route 1 → grass,
accumulating ~50-200 of EXPLORE reward per episode (depending
on tiles covered). Wild battles after that lose at -25 each, but
the explore reward dominates the mean across 24 envs.

Bimodal distribution per episode:
- "Won Blue + explored": +200 to +300
- "Lost Blue or died on Route 1": -25 to -40

60/40 split of those buckets averages ~+158. The agent learned
exploration but not in-battle decision-making.

### Three hypotheses for "why Tail Whip"

There has to be positive reinforcement on Tail Whip; otherwise
the policy would settle near 50/50. I sat with this for a while
and the structural answer surfaced:

**1. Removing STEP_PENALTY made battle-menu existence free.** This
is the load-bearing hypothesis. In V0.3.x, STEP_PENALTY = -0.001/step
made "stand still" mildly costly. V0.4 set it to 0 along with the
other strip. Now during Tail Whip turns, the agent advances frames
(animations, HP bar updates, menu reappears) and pays *nothing*.
Tackle ends the battle, cycling back to the overworld where
another battle is probably waiting to kill you. Tail Whip *stays
in the safe menu*. Discounted PV of "Tail Whip and not die for 20
more steps" exceeds "Tackle, finish, walk into next likely-fatal
battle."

**2. Menu cursor inertia.** Pokemon Red's FIGHT menu remembers the
last-used move. Once the agent randomly Tail Whips once, "press A"
reselects Tail Whip without the agent learning anything about
*which* move it's picking. The downsampled grayscale obs doesn't
clearly show the small cursor sprite, so the policy can't
differentiate "cursor on Tackle" from "cursor on Tail Whip" — it
just learned "A in battle = neutral thing happens." Compounds with
#1: once cursor is stuck on Tail Whip, no gradient pressure to
move it.

**3. Discount-rate effect.** PV(-25 at step 5) ≈ -24.9 vs PV(-25
at step 20) ≈ -24.5. Delaying the eventual blackout shaves
fractional return. Probably second-order to #1 and #2.

### The diagnostic asymmetry

Important: the failure isn't "V0.4 needs per-action reward signal
to discriminate Tackle from Tail Whip." It's narrower than that.
V0.4 inadvertently made *being in a neutral state forever* a
winning strategy by removing all per-step friction. The reward
landscape rewarded HP gains (heal_quad) but was silent on HP
losses. Tail Whip = enemy hits Squirtle every turn = sustained
unrewarded damage. The reward function had no way to *see* the
damage.

### V0.4.1: DAMAGE_QUAD symmetric to HEAL_QUAD

The fix: mirror the heal reward on the damage side.

```
delta_hp_frac < 0  ->  reward -= (delta)^2 * 15
```

Same coefficient (15) as HEAL_QUAD. (delta)² weighting matches
the heal curve. Per-turn damage of ~15% HP pays roughly -0.34.

Math check:
- Tackle-win 4-turn battle: ~-1.4 total damage tax
- Tail Whip 10-turn stall: ~-3.4 total damage tax

Tackle becomes mathematically preferable without us adding any
per-action reward. The agent should discover this through the
existing TRAINER_WIN_BONUS + falling per-area kill log + the
new asymmetric damage cost.

Faint events excluded (`cur == 0`): FAINT_PENALTY -5 already
handles those, no double-billing.

### Why this still respects the V0.4 thesis

The V0.4 meta-discipline (Day 3) was: don't add knobs to map
direct behavior; restructure the landscape. DAMAGE_QUAD is *not*
"penalize Tail Whip." It's "every state where HP went down is
slightly worse." It rewards state changes, not actions.

The cleaner framing: V0.4 was internally asymmetric — heal
rewarded, damage silent. V0.4.1 restores symmetry. Still
curve-shaped, still state-based, still future-seeking.

I'm willing to ship this without violating the discipline because
the fix isn't behavioral shaping — it's making the existing
HEAL_QUAD reward two-sided.

### Entropy sweep findings (negative)

Along the way I also tried higher-entropy variants of V0.4: e03
(0.03) and e05 (0.05), running alongside e01 (0.01). All three
showed Tail Whip lock-in to varying degrees. The hypothesis that
higher entropy would let Tackle stay in the in-battle sampling
distribution long enough to discover wild-battle wins didn't hold
— the structural problem (no in-battle damage signal) dominates
the entropy effect. Cancelled e03/e05 when V0.4.1 shipped; e01
(17653) kept running as the no-DAMAGE_QUAD control.

### What's running tonight

- **17653** — V0.4 baseline (entropy 0.01, no DAMAGE_QUAD)
- **17725** — V0.4.1 (entropy 0.01, DAMAGE_QUAD = +15)

Same scale (3 GPUs each on different L40S nodes), same config
except reward class. Clean A/B by morning.

### Operational change

Updated repo CLAUDE.md to be a primer for new agent sessions:
pointers to design-log, vault MOC, decision notes, ELSA workflows,
and the discipline. Also softened the "4-GPU is dead" framing to
"check cluster state before submitting" — 4-GPU availability is
intermittent, not gone. Added concrete queries for picking the
right GPU count by current node state.

### Open thread

The 0.4.1 fix tests the "no per-step friction" hypothesis. If
17725 climbs past 17653's plateau by morning, that hypothesis
landed. If both flatline at the same plateau, the real issue is
the menu-cursor-inertia hypothesis (#2 above) and we need a
different fix — probably action-masking in the battle menu, or
including the cursor position in the obs more explicitly.

Tomorrow's read tells us which hypothesis was load-bearing.

## 2026-06-05 — Day 5: V0.4.1 menu-stall + V0.4.2 three-arm design + cluster block

### How the morning read fell

Watched 17653 (V0.4 baseline) first at iter 16100 — ~3× longer
than the original iter 5000 Tail Whip diagnosis. Still spamming
Tail Whip in wild battles. So the "more steps will fix it"
possibility is dead: asymmetry was load-bearing, not training
duration. V0.4 baseline is structurally stuck.

Pulled metrics on 17725 (V0.4.1) at iter 12600 (~77M steps):
mean_return_last20 = **-19.5**, entropy = **1.10–1.17**,
unique_tiles_total = **24**. The combination is a much louder
signal than I expected. Entropy near 1.1 at iter 12600 with
entropy_coef=0.01 is well above committed-policy levels (max is
log(7) ≈ 1.95). Return is negative. Tile count is 24 — agent
isn't progressing past the Blue rival battle at all.

### What V0.4.1 actually learned

Initially I read the metrics as "DAMAGE_QUAD over-deterred
combat, agent refuses to fight." Then Colin watched it. The
agent **knows it can't run from a trainer fight**, so it's not
just refusing — it's actively navigating menu state to avoid
committing a move. Opens FIGHT → backs out → opens PKMN → backs
out → opens ITEM → backs out. Mashing B and cancelling out of
submenus indefinitely. The enemy never gets a turn (you can't
attack until the player commits a move), so Squirtle takes no
damage and the DAMAGE_QUAD tax never fires.

DAMAGE_QUAD did exactly what we asked: the agent learned that HP
loss is bad. It just routed around the signal via a path I hadn't
modeled. The Day 4 menu-cursor-inertia hypothesis (#2) is now
real — not as a hypothesis about V0.4 baseline but about V0.4.1's
fix. The agent learned to avoid the cursor position that triggers
the enemy turn.

### The structural pattern (and what V0.4 actually broke)

Both V0.4 baseline (Tail Whip lock-in) and V0.4.1 (menu-stall)
are symptoms of the same structural hole: **V0.4 removed all
per-step in-battle friction.** Without that friction, whichever
in-battle state pays the least cost becomes the attractor.

Under V0.4: Tail Whip pays nothing per turn (DAMAGE_QUAD doesn't
exist, STEP_PENALTY removed), so the agent parks there. Tail
Whip happens to also pay nothing in damage to the enemy, and the
enemy eventually KOs Squirtle (faint -5, lose -20), but the
discounted-PV math makes "stay in the safe menu" preferable.

Under V0.4.1: DAMAGE_QUAD made FIGHT costly relative to non-FIGHT
menu navigation, so the agent now parks in submenu navigation
where no enemy turn ever happens. Same structural pattern, new
free-state attractor.

The Day 4 "restore STEP_PENALTY" rejected alternative needs to be
revisited with this evidence. It was rejected as "too blunt;
penalizes all menu turns including legitimate strategic ones."
That objection still stands for a global STEP_PENALTY. But it
doesn't stand for an in-battle-scoped activity-based penalty —
the legitimate use of in-battle menu time is short
(single-deliberation turns), and a triggered penalty rather than
a flat per-step penalty preserves that.

### V0.4.2 — three-pronged structural fix

Three things change together:

**1. BATTLE_STALL: -0.01/step after 30 consecutive steps of no HP
change on either side.** This is STUCK ported to battle. STUCK
fires when no new tile is visited for 200 steps — it's
activity-based, state-shaped, scoped to the "you've been in the
same situation too long" regime. The current STUCK never fires
in battle because tile count doesn't change in battle. The battle
equivalent is no HP delta on either side. Same philosophy, same
gentle slope, scoped to in-battle frames.

Menu-stall → no HP change → counter increments → after 30 steps,
penalty fires every step.
Tail Whip lock-in → enemy attacks every turn → HP delta → counter
resets → no BATTLE_STALL, but DAMAGE_QUAD fires.
Tackle exchange → HP delta from both sides → counter resets,
BATTLE_STALL doesn't fire, DAMAGE_QUAD partial.

Tackle becomes mathematically preferable without any move-specific
knob. The agent retains freedom to use Tail Whip strategically.

Coefficient calibration: STUCK is -0.005/step. Doubled to -0.01
because in-battle stall has zero incidental value (overworld
wander loops might still trip new tiles incidentally; menu-stall
gets nothing).

**2. DAMAGE_QUAD_COEF: 15 → 5.** Asymmetry signal preserved
(Tackle still beats Tail Whip by ~2-3× cumulative damage tax) at
a magnitude that doesn't dominate bootstrap. Per-turn at ~15% HP
loss: -0.34 → -0.11, roughly matches LEVEL increment magnitude.

The original 15 was chosen for state-shape symmetry with HEAL_QUAD
(also 15). But the firing-frequency asymmetry makes the
magnitudes effectively asymmetric in the other direction:
HEAL_QUAD fires rarely (heal events sparse), DAMAGE_QUAD fires
every turn of every battle. Matching coefficients gave a cumulative
damage tax that swamped the rest of the landscape.

**3. LOSE_BATTLE: -20 → -10.** V0.4's raise from -7 to -20
predated DAMAGE_QUAD. With DAMAGE_QUAD now handling ongoing damage
cost during losing fights, -20 was double-counting the same fight.
-10 keeps blackout meaningfully costly without that
double-counting.

### Bootstrap EV math (the load-bearing argument)

The reason V0.4.1 cratered isn't that any single coefficient was
wrong. It's that the combined negative outcome of "lose a fight"
under V0.4.1 was -28 (~-3 damage tax + -5 faint + -20 lose),
while the positive outcome of "win a fight" was only ~+8 (after
~-5 damage tax during the win). At any bootstrap-realistic win
rate (~30% for an uncertain Squirtle vs Bulbasaur with type
disadvantage), expected return per battle was -17. That matches
observed -19.5. The Blue rival fight was genuinely net-negative,
so the agent's correct response was "don't fight."

Under V0.4.2 (center arm, DQ=5, LOSE=-10):
  Win: BEAT_MON +1.96, LEVEL +0.1, TRAINER_WIN +10, DAMAGE_QUAD ~-1
       = +11
  Lose: DAMAGE_QUAD ~-1, FAINT -5, LOSE_BATTLE -10 = -16
  EV at 30% win = 0.3 × 11 + 0.7 × -16 = -7.9 (was -17.2)

Breakeven win rate: 57% (was ~75% in V0.4.1). Within
bootstrap-recoverable range — the agent only needs one strategy
that crosses 60% to start the learning loop.

### Three arms, spread by breakeven

Submitted as a sweep across the bootstrap-feasibility range:

| Variant | DAMAGE_QUAD | LOSE_BATTLE | Breakeven | Scale | Entropy |
|---|---|---|---|---|---|
| **center** | 5 | -10 | ~57% | ddp3 (node004) | 0.01 |
| **harsh** | 8 | -15 | ~64% | ddp2 (node002) | 0.005 |
| **gentle** | 3 | -7 | ~51% | 1gpu (node003/005) | 0.005 |

Center gets the 3-GPU allocation as the expected-best outcome
with standard entropy. Harsh and gentle are diagnostic probes at
eager entropy — they're meant to commit fast so we can read what
each landscape favors at iter 3000-5000.

What the spread tests:
- If center works and gentle works too → landscape is robust;
  damage magnitude was the dominant axis
- If only center works → calibration is narrow; we'd need finer
  sweeps
- If only gentle works → DAMAGE_QUAD=5 is still too harsh for
  bootstrap; need to keep dropping
- If only harsh works → BATTLE_STALL is doing all the work and
  damage calibration was secondary
- If none work → menu-cursor-inertia is structural beyond
  reward design; need action-masking or richer obs

### Why three arms now (not one at a time)

Normally I'd ship one variant and iterate. Three reasons for the
spread this time:

1. V0.4.1's data already pre-falsified DAMAGE_QUAD=15 +
   LOSE=-20 as too harsh. So we're not testing those
   independently — we're testing whether the recalibrated range
   has a sweet spot.
2. The three failure modes are different enough (Tail Whip
   lock-in, menu-stall, refusal) that getting all three answers
   from one variant would still leave ambiguity. The spread
   isolates which axis is load-bearing.
3. ELSA had 7 free GPUs at submit time. Three single-node runs
   fit without queue contention. The marginal cost of three vs
   one was just submission time.

### Cluster block (operational)

After submitting the three V0.4.2 sbatches, all three sat in PD
with `QOSMaxCpuPerUserLimit`. Cancelling the two V0.4 grandfather
jobs (17653, 17725) did not unblock them — the new submissions
then hit `QOSMaxGRESPerUser`. Checked the QOS config:

```
sacctmgr show qos starter
  MaxTRESPU = cpu=4,gres/gpu=0,mem=32G
  MaxJobs = 4
  MaxSubmit = 10
```

My account is on `starter` QOS, which literally allows zero GPUs
and 4 CPUs per user. The previous days' 3-GPU ddp3 jobs were
grandfathered through under what must have been a less-restrictive
prior QOS state. Anything new submitted today is hard-blocked.

This is a cluster-side policy change, not anything we did. Most
likely either (a) routine QOS migration that landed on my account,
or (b) fairshare-tier downgrade after sustained heavy usage over
3 days. Email sent to Sean Sivy (cluster escalation contact)
asking whether this is misconfig or appropriate throttling, and
offering to reduce to single-GPU runs / off-peak windows if
needed.

V0.4.2 code is committed (74fcf61), pushed, ready on ELSA. Three
sbatches will submit cleanly once the QOS unblocks. Until then,
training is paused.

### Operational note for future me

Cancelling running jobs to free quota for new submissions only
works if the new submissions are *eligible* under your current
QOS. Always check `sacctmgr show qos <yourqos>` before scancel —
if the QOS itself has been tightened, freeing CPUs doesn't help.
The 17653/17725 cancels lost actual training data (they would have
kept running indefinitely) for no scheduling gain.

### Open thread

When ELSA unblocks: submit the three V0.4.2 sbatches, watch each
at iter 3000-5000 (low-entropy probe pattern), interpret per the
"what the spread tests" framework above. If center or gentle
land on Tackle and clear the bootstrap, V0.4.2 was the right fix.
If all three fail in different ways, that's strong evidence the
in-battle problem is beyond reward design and we need to look at
the observation/action space (cursor visibility, action masking).


## 2026-06-09 — Day 6: VCL pivot, async hang root cause, infra wins

### The gap

Day 5 (6/5) ended with V0.4.2 staged and the QOS block in place.
Email to Sean Sivy went out the same afternoon. Nothing happened
on the cluster front for the next three days. I didn't fight it,
mostly because I was burned out on infra after the V0.4 → V0.4.1 →
V0.4.2 sprint, and partly because the right move was to wait for
Sean to get back rather than escalate.

Today (6/9) Sean responded and Brad Mott (the RL contact Jessica
suggested at NCSU) also responded. Two parallel emails, both
moving things forward.

### Brad's reply (RL guidance)

I'd asked Brad about the state-based curve approach (the V0.4
philosophy shift away from action-specific rewards). His response:
the move toward broader state-based signals is sensible, and the
main thing to watch for is one reward component dominating the
others. He pointed me at the IEEE Conference on Games proceedings
for similar work.

The reward-dominance warning is exactly what's been biting me —
flee-as-win dominated V0.2.1, walking-left dominated V0.2 entropy
collapse, Tail Whip dominated V0.4. Each one was a different
component winning the gradient race. Hearing it framed as a known
RL failure mode rather than my own confusion was useful.

Brad also approved using NCSU's VCL for the Pokemon project while
the TCNJ cluster is blocked. This was the actual unblock — it
meant I had a path to keep training without waiting for Sean.

### Sean's reply (cluster ops)

Sean confirmed the QOS profiles are being restructured per-lab /
per-class. He asked which faculty member I'm working with on this
project. The honest answer is: I do research with Dr. Yoon, but
the Pokemon RL work is personal portfolio, not a lab project.
Another faculty member had told me informally that personal use
of ELSA was fine, but that wasn't an official sponsorship. I told
Sean that directly in the reply and let him decide whether it
fits the new framework.

He also flagged two operationally useful things:
1. My L40S jobs were using <1GB VRAM. He suggested the GTX 1080Ti
   nodes (3 × 8 GPUs, 11GB each, infrequently used). This is
   correct — the bottleneck is PyBoy CPU, not VRAM. 11GB is plenty
   for a tiny policy network.
2. A pre-empt queue if my code supports check-pointing. I said
   yes (periodic model saves exist) and we'd follow up on whether
   a proper SIGTERM handler is in place before committing.

Net: ELSA is potentially unblocking on 1080Ti + pre-empt, but
that's a few days out at best. VCL is available now.

### Pivot to VCL (and why VCL specifically)

VCL is NC State's Virtual Computing Lab — reserve a VM, get a
GPU + CPU for up to 10 hours per reservation. The image I
reserved was Ubuntu 22 + CUDA + RTX 2080 Ti. 16 cores, 88GB RAM,
11GB VRAM.

Why VCL over alternatives:
- Brad gave explicit permission. The institutional question is
  resolved without ambiguity.
- Free for NCSU students. Local-first ML rule still holds.
- Roughly the same single-GPU specs as ELSA's 1080Ti suggestion
  Sean made, but available right now instead of in a few days.
- The 10-hour reservation limit is a real constraint but
  workable with checkpointing.

Tradeoffs accepted:
- Single node only — no DDP, no multi-GPU. Per the Day 2 multi-
  node analysis this is fine; 4xL40S only got 1.47x over single
  L40S because PyBoy is the bottleneck.
- VCL is a desktop VM with Xorg running. This turned out to
  matter — see the async hang section below.
- No SLURM — runs are direct python invocations in tmux. The
  sbatch infrastructure doesn't translate.

### Setting up the repo on VCL

uv install worked clean. The repo synced via rsync from local
since the github repo is private and the VCL VM didn't have
credentials. State files (`states/blue_fight.state` etc.) had to
be rsynced separately because they're gitignored.

The smoke test (sync envs, 2 envs) ran fine at ~130-146 sps and
confirmed the env code worked unchanged on VCL.

### Branch strategy: vcl, elsa, main

Created a `vcl` and `elsa` branch off main. The motivation: the
two cluster contexts have different operational shapes (SLURM vs
direct invocation, multi-GPU configs vs single-GPU, sbatch
scripts vs tmux launchers). Trying to keep all of it on main
would bloat the configs/ directory and confuse the next session.

On the `vcl` branch:
- Removed `slurm/` directory entirely
- Removed all `elsa_brock_*` configs
- Removed `sync_from_elsa.sh`
- Added `setup_vcl.sh` (one-time env install), `launch_vcl.sh`
  (tmux + uv wrapper), `sync_from_vcl.sh` (pull runs back local)
- Replaced README with a VCL-focused version
- Only `vcl_*` and `dev_local` configs remain

`elsa` branch keeps the existing infrastructure as-is. `main` is
shared code (envs, rewards, PPO). Reward changes land on main
first, then merge into whichever cluster branch is active.

This is heavier than necessary for one project, but I expect this
"two cluster contexts" pattern to recur (REU has VCL, TCNJ has
ELSA, future projects will have something else). The branch
split makes the operational context explicit.

### First run on VCL (sync envs, slow but working)

V0.4.2 center arm, n_envs=12 sync, frame_skip=1. ~150-175 sps
after JIT warm-up. About half the throughput of single-GPU ELSA
async, but enough to make real progress.

Got to iter 300 by 20:13 (about 90 minutes of training). Pulled
the iter 300 checkpoint and watched it locally. Behavior was
healthy for an early checkpoint:

- Resolved the rival battle (didn't get stuck in menu loops)
- Long exploration phase in Oak's lab pressing A on objects
- Eventually navigated through the lab door
- Triggered a wild encounter in the grass after leaving

No obvious degenerate behavior. BATTLE_STALL didn't visibly break
anything. The "presses A on everything" pattern is mild noise
from H=1.92, not a pathology. Iter 300 is far from the real
diagnostic window (iter 3000-5000) but the sanity check passed.

### Async envs hung — the actual story

The original config used `async_envs: true` because that's what
ELSA used. On VCL it just hung forever. 12 worker subprocesses
spawned and consumed 50% CPU each, but the parent process never
received the first observation.

First thing I checked was `/dev/shm` — the standard PyTorch
multiprocessing IPC channel, often undersized on VMs. It was 45GB.
Not the issue.

CUDA initialization order was the next suspicion — the standard
trap is fork-after-CUDA-init. Checked the train.py + ppo.py
ordering: `env_fn()` runs on line 128 of train.py, but CUDA
device + ActorCritic.to(device) doesn't happen until ppo.py
line 148+. So at the moment workers spawn, CUDA has not been
touched in the parent. Not the issue either.

The clue was the workers being at 50% CPU but the parent receiving
nothing. The workers were doing work, they just weren't
communicating it back. That pointed at something blocking
during worker startup before they reached the IPC loop.

Looked at PokemonRedEnv `__init__`: PyBoy is created with
`window="null"` (headless), so no SDL2 window should open. But
pysdl2-dll is imported at module level (the warning is visible
in every log). On VCL the host has Xorg running because it's a
desktop image. The hypothesis: SDL2 in subprocess context tries
to connect to the X display for initialization even when PyBoy
doesn't open a window, and on the VCL VM that connection blocks
or fails silently.

Fix: set `SDL_VIDEODRIVER=dummy` in the launch env. This forces
SDL2 to use a no-op video driver that doesn't touch X11.

Smoke test (2 envs async, SDL_VIDEODRIVER=dummy): clean, ~210-249
sps. Ran the full 12-env async config: **963 sps at iter 10**.
About 6x the sync throughput.

### Why this matters operationally

The async fix turned VCL from "barely fits a meaningful run in
10 hours" into "diagnostic window in ~90 minutes." At 963 sps
with 3072 steps per iter:

- iter 100: ~5 minutes
- iter 1000: ~50 minutes
- iter 3000 (lower bound of diagnostic window): ~2.5 hours

That's well inside the 10-hour reservation budget. The async fix
is the difference between "useful for diagnostic" and "barely
usable for sanity check."

For comparison: the old ELSA DDP4 metrics from V0.2 show ~1780
sps. CLAUDE.md notes DDP4 was only 1.47x over single L40S, so
single L40S async was roughly 1210 sps. VCL 2080 Ti async at 963
sps is in the same order of magnitude as ELSA single-GPU async,
not catastrophically slower. The 2080 Ti is a meaningfully
weaker GPU than the L40S, but PyBoy bottleneck dominates so the
GPU spec gap doesn't show up at full scale.

### Frame skip code (added, not yet enabled)

Per the optimization plan, I added a `frame_skip` parameter to
PPOConfig and PokemonRedEnv. The env's step() now repeats the
24-frame button-press loop N times before reading reward. Default
is 1 (current behavior). Tests pass.

I didn't enable frame_skip on the resume run because the iter 300
policy was trained with frame_skip=1; switching mid-training
changes the effective env from the policy's perspective. Cleaner
to test frame_skip on a fresh run.

Next chance to validate frame skip: probably a separate VCL run
or after the resume run finishes.

### Why no Gambatte backend

The plan included a third optimization: replace PyBoy with
Gambatte (C++ Game Boy emulator, much faster per tick). Researched
it before committing. Findings:

- No clean pip-installable Python binding for Gambatte exists.
  The two viable paths are `stable-retro` (broken on Apple Silicon
  for Game Boy core) and `pdretro` (new, single author, RAM
  reads unconfirmed).
- No existing Pokemon RL project has switched off PyBoy. PufferAI's
  pokegym hits hundreds of thousands of sps with PyBoy across
  parallel envs.

Conclusion: scope Gambatte out. Multi-day yak-shave for unclear
additional gain once async is fixed. The async fix was the real
unlock; the per-instance emulator speed wasn't the binding
constraint.

### Resuming from iter 300

After confirming the async fix worked, killed the new from-scratch
async run and relaunched with `--resume runs/.../iter_000300.pt`.
This loads the policy weights from iter 300 (the watched
checkpoint) and continues training with the faster setup.

The new run is logging to `runs/vcl_brock_v0_4_2_center_1gpu_resume/
train.log`. iter 10 of the resume showed mean_return +22.67 vs
+5.77 in the earlier sync run at iter 170 — the policy from iter
300 is meaningfully more committed than what we'd been seeing at
iter 170.

### Operational decisions / why

A few changes that went into the project today, briefly justified:

1. **`SDL_VIDEODRIVER=dummy` in launch_vcl.sh**: Forces SDL2 away
   from X11. Required on VCL because of the Xorg desktop. Won't
   hurt anywhere else (ELSA doesn't have an X server to connect
   to in the first place, so the env var is a no-op).
2. **`PYTHONUNBUFFERED=1` in launch_vcl.sh**: First sync smoke
   test produced no output for minutes because Python buffered
   stdout. Forced unbuffered output so the launch log is
   immediately useful for debugging.
3. **`async_envs: true` re-enabled in VCL config**: Now that the
   SDL fix is in place, no reason to stay on sync. The 6x
   throughput is worth more than the slight cognitive overhead
   of remembering the env var.
4. **`frame_skip: int = 1` field on PPOConfig**: Adds the
   parameter without changing default behavior. Lets us A/B test
   frame skip on a fresh run without risking the resume.
5. **Periodic check via cron**: 30-min checks set up while away
   from the machine. Reports the latest iter / sps / mean_return
   / entropy. Catches process death or obvious regression before
   the next time I'm back at the machine.

### Open thread

- Wait for the resume run to hit iter 3000-5000 (diagnostic
  window) and watch a checkpoint. That's the actual V0.4.2
  verdict: did BATTLE_STALL fix the menu-stall, did the
  recalibrated DAMAGE_QUAD produce committed combat, or did
  something else come up.
- Validate frame_skip on a fresh run once the resume is past
  the diagnostic window. If frame_skip=4 gives clean 4x on top
  of the 963 sps baseline (target: ~3000-4000 sps), that's the
  emulator-bottleneck fix and matches single L40S DDP4 throughput
  on a single 2080 Ti.
- Sean still owes a final answer on whether the personal-project
  use of ELSA fits the new QOS framework. Whatever he says, the
  vcl branch is now a real fallback so ELSA being intermittent
  is no longer blocking.
- Reward-version writeup (`notes/reward-versions.md`) is still
  the canonical portfolio source. Today's work doesn't change
  the reward arc, just the infrastructure. If anything, the
  pivot story is a worth-including chapter in a "what I learned
  doing this" section of the writeup: production RL is mostly
  infra debugging, and one env var stood between "barely usable"
  and "throughput parity with the institutional cluster."
