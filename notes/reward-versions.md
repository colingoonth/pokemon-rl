# Reward function evolution

Each version, its design rationale, and what training revealed. Source
material for the writeup's "what I learned about reward design" section.

Naming: V0.x are exploratory iterations. **V1.0 is reserved for the
reward function that actually clears Brock.**

---

## V0.1 — first honest attempt (cancelled, env-bug confounded)

**File:** `RewardV0_1` in `pokerl/env/rewards.py`

**Components and weights:**

| Signal | Weight |
|---|---|
| New `(map_id, x, y)` tile | +1 |
| Total party levels gained | +5/level |
| New event flag | +10/flag |
| Badge | +100 |
| Step penalty | -0.001 |

**Design rationale.** Matched my pre-RL gut instinct (exploration + combat
progress + story milestones) onto a minimal reward function. The categories
came from my own thinking; the weights were a guess at "this seems
proportional." Intent was a baseline to compare iterations against.

**ELSA run summary.** Trained 32 envs × N hours.

**Observed behavior:** `tiles(all_envs)` stuck at 32 (each env visited
exactly its spawn tile), entropy hovering near max, mean return locked
at the step-penalty floor of -4.096 per truncated episode.

**Diagnosis.** Looked like the policy never learned to press movement
buttons. BUT — running a deterministic "press up 30 times" test on a
single env locally showed the player ALSO never moved.
**The env was broken**: `pyboy.button(name)` was a 1-frame press, which
Pokemon Red is too slow to register as a movement command. V0.1's training
was doomed before reward design mattered. Fixed by using
`button_press` + hold for 12 frames + `button_release` pattern in
`pokemon_red_env.py`.

**Lesson recorded.** Verify env mechanics deterministically before
trusting any conclusion drawn from policy training. The reward design
was not the bug, the env was.

---

## V0.2 — battle awareness + movement bonus (after env fix)

**File:** `RewardV0_2`

**Components and weights:**

| Signal | Weight | Notes |
|---|---|---|
| New `(map_id, x, y)` tile | +1 | Carried from V0.1 |
| Movement to visited tile | +0.2 | NEW — credit any position change, not just new tiles. Combats "policy refuses to move because moving doesn't immediately help." |
| Step penalty | -0.001 | Carried |
| Total party levels gained | +3/level | Down from +5 (tuning) |
| New event flag | +10/flag | Carried |
| Badge | +100 | Carried |
| New encounter (new species or new trainer) | +5 | NEW — battle engagement signal |
| Beat an enemy Pokemon (HP -> 0) | +2 | NEW — granular combat reward |
| Party member faints | -2 | NEW — discourages losing |
| Win battle (party still alive at battle end) | +10 | NEW — successful combat |
| Lose battle (all party fainted) | -7 | NEW — blackout penalty |

Move bonus does NOT fire during `in_battle != 0` (dialogue/battle locks
position) so the agent isn't punished for engaging with NPCs.

**Design rationale.** The post-fix env enabled real exploration, but
exploration alone wouldn't reach Brock. Added a full combat reward
suite to give the agent a path through battles toward leveling +
badges.

**ELSA run summary.** Job 13727. ~1 hour, ~1190 iterations.

**Observed behavior.** Trained beautifully for the first ~30 iterations:
tiles climbed from 32 → 5525, entropy dropped 1.94 → 1.78, mean return
jumped to +707. Then **collapsed**: by iter 1100, entropy = 0.0, tiles
back down to 384, mean return stuck at +6.9. Brief flickers of recovery
(iter 1190 showed entropy 0.59 + tiles 422 + return +38.7) suggested
the agent occasionally broke out of the local optimum when forced into
an unfamiliar state.

**Diagnosis.** Classic PPO **entropy collapse**. The exploration reward
was easy to grind by walking in a small deterministic loop (revisits
pay +0.2 = 200× the step penalty). The policy converged on a low-entropy
"walk in this one direction" strategy that maximized reward without
exploring action space. Confirmed by watching the iter_001100.pt
checkpoint locally: agent stuck at first wall, when it slipped through
it entered a Rattata battle but never advanced past "Wild RATTATA
appeared!" because the deterministic policy didn't press A.

**Lesson recorded.** With sparse-but-grindable signals + low entropy
regularization (PPO's `entropy_coef=0.01`), a learnable degenerate
strategy will dominate. Either reduce the grindable signal magnitude
or strengthen entropy regularization (or both).

---

## V0.2.1 — exploration scaled down 3×

**File:** `RewardV0_2_1` (inherits from `RewardV0_2`)

**Changes from V0.2:**

| Signal | V0.2 | V0.2.1 |
|---|---|---|
| New tile | +1 | **+0.3** |
| Move to visited tile | +0.2 | **+0.05** |

All other weights inherited unchanged.

**Design rationale.** Hypothesis: V0.2 collapsed because exploration
rewards dominated, making "walk in a circle" too rewarding. Scaling
them down ~3× should make combat rewards (+5 encounter, +10 win) more
attractive relative to easy exploration. Did NOT touch PPO's
`entropy_coef`, so this isolates the reward-scaling lever from the
entropy-regularization lever.

**ELSA run summary.** Job 13946. ~1 hour, iter ~880.

**Observed behavior.** Compared to V0.2 at similar training horizon:

| | V0.2 (iter 1100) | V0.2.1 (iter 880) |
|---|---|---|
| Entropy | 0.0 (collapsed) | **0.95-1.25** (healthy) |
| Tiles | 384 (stuck) | **1700-2100** (growing) |
| Mean return | +6.9 (stuck) | **+1034** |

**Reward scaling alone was enough to prevent the entropy collapse.**
Entropy stayed in the 1.0-1.25 range — the policy formed strong
preferences but didn't go fully deterministic.

But then a different problem surfaced from watching `iter_000800.pt`:
**the flee-as-win exploit.** Agent walked into grass, encountered
Rattata, immediately fled, in_battle went from 1 → 0 with the party
still alive, +10 WIN_BATTLE fired. Repeat. Free +10 per encounter
once species-reward exhausted. Visible in the watch: the policy was
learning "encounter, flee, encounter, flee" as a grind strategy.

**Diagnosis.** Bug in my V0.2 implementation:
```python
if party_alive:
    reward += self.WIN_BATTLE
```
Was meant to credit kills/catches, instead credited any battle exit
where the party survived — including fleeing.

**Lesson recorded.** A reward signal that the spec says "fires when X
happens" must be implemented to only fire when X actually happened.
"Battle ended with party alive" is NOT "player won the battle."
Detection needs explicit state tracking.

---

## V0.2.2 — fix flee-exploit + new-map + catch (running / queued)

**File:** `RewardV0_2_2` (inherits from `RewardV0_2_1`)

**Changes from V0.2.1:**

| Signal | V0.2.1 | V0.2.2 |
|---|---|---|
| Beat enemy Pokemon (HP -> 0) | +2 | **+10** (this IS the wild-win reward now) |
| Generic WIN_BATTLE on battle end | +10 always | **removed** (replaced by per-kill + trainer bonus) |
| Trainer battle win (kill at least 1 mon) | implicit in +10 | **+20 bonus** on top of per-kill rewards |
| Flee from battle | +10 (bug!) | **-1** (FLEE_PENALTY) |
| Catch a Pokemon (party_count++) | not rewarded | **+5** (CATCH_REWARD) |
| Catch a new species | not rewarded | **+20 bonus** (NEW_CATCH_BONUS) |
| Enter a new map_id | not rewarded | **+5** (NEW_MAP_REWARD) |

**Design rationale and reward math.**

The big behavioral targets:
1. **Stop the flee exploit.** Only kills and catches count toward the
   "successful battle" signal. Track `enemy_killed_this_battle` and
   `pokemon_caught_this_battle` during the battle.
2. **Encourage fighting over fleeing.** Killing a wild Pokemon now
   pays +10 directly (vs. -1 for fleeing). Strategic flee still
   cheaper than -7 lose + faint penalties.
3. **Trainer battles weighted higher than wild.** Trainer = +20 bonus
   on top of per-kill +10. Defeating a 3-Pokemon trainer = 3×10 + 20 = +50.
4. **Encourage party-building, but not random catching.** Catching
   pays +5 base + +20 only for new species. Duplicates pay only +5,
   way less than the +10 kill reward, so catching duplicates is
   disincentivized.
5. **Pull the agent out of "grind battles forever."** Once the agent
   converges on combat, it needs a reason to enter Viridian's PokeMart,
   PokeCenters, Brock's Gym, etc. The +5 per new map_id is a sparse
   macro-exploration signal that fires every time the agent enters a
   new building or route.

**Reward math at the new weights:**
- Kill wild Pidgey: **+10**
- Catch new Pidgey: **+5 + +20 = +25**
- Catch duplicate Rattata: **+5**
- Trainer battle, kill 3 Pokemon: **+30 + +20 = +50**
- Flee from any battle: **-1**
- Lose battle (whole team fainted): **-7 + faint penalties**
- Enter Viridian City for first time: **+5**

**Hypotheses to test in ELSA training:**
- Does the flee exploit go away? (Should — code change is direct.)
- Does the agent learn to fight wild Pokemon rather than flee?
- Does the agent enter buildings / new maps to chase the +5 bonus?
- Does the agent catch Pokemon — particularly new species — to chase
  the +25 jackpot?
- Does entropy stay healthy (>0.5) like in V0.2.1, or does some
  newly-grindable signal cause another collapse?

**Status:** Config + sbatch ready
(`configs/elsa_brock_v0_2_2.yaml`, `slurm/train_brock_v0_2_2.sbatch`).
Launch when V0.2.1 is cancelled.

---

## Tracker for future versions

| Version | Status | Headline change | Outcome |
|---|---|---|---|
| V0.1 | Cancelled | First attempt | Confounded by env button-hold bug |
| V0.2 | Cancelled | + movement bonus + battle suite | Entropy collapse to 0 by iter 1100 |
| V0.2.1 | Cancelled | exploration scaled 3× down | Avoided collapse but exposed flee-as-win exploit |
| V0.2.2 | Queued | flee fix + new-map / catch rewards | TBD — pending run |
| V1.0 | Reserved | First version to clear Brock | — |
