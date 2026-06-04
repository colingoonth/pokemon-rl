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

**ELSA run summary.** Job 14143. Cancelled at iter 800 (~1 hour, ~6.4M steps)
once the blackout exploit was identified from watching `iter_000800.pt`.

**Observed behavior.** Entropy held healthy (no collapse). Tiles per batch
high (4.5k–20k). Mean return last20 oscillated +230 to +390 — looked good
on paper. But watching the policy revealed: agent explored aggressively,
let its starter faint, blacked out, respawned fully healed, kept going.
Never voluntarily used a Pokemon Center.

**Diagnosis.** The **blackout-as-free-heal exploit**. V0.2.2 keeps the
inherited `FAINT_PENALTY = -2` and `LOSE_BATTLE = -7`. Net cost of dying:
about -9. Per-life exploration gain: ~+200-300. Net per-blackout-cycle
is strongly positive, so the policy converged on "die, respawn, explore."
Same structural shape as V0.2.1's flee-as-win — a "loss" event netting
positive reward because the alternative routes pay less.

**Lesson recorded.** Whenever a "loss" event has a partial-refund
mechanic in-game (whiteout heals, flee escapes), the penalty must
dominate the alternative path's reward — not just be "non-trivial."

---

## V0.2.3 — Pokemon Center heal rewards + harsh faint penalty

**File:** `RewardV0_2_3` (inherits from `RewardV0_2_2`)

**Changes from V0.2.2:**

| Signal | V0.2.2 | V0.2.3 |
|---|---|---|
| Party member faints | -2 | **-250** |
| Heal at PC, low HP (`party<80%` OR `any<50%`) | not rewarded | **+2** (PC_HEAL_LOW) |
| Heal at PC, near-full HP | not rewarded | **-1** (PC_HEAL_FULL — anti-spam) |
| First-time visit to a given PC map_id | not rewarded | **+5** (PC_FIRST_VISIT, stacks) |
| Blackout-triggered heal/visit | n/a | **suppressed** (no refund of -250) |

PC map_ids hardcoded in `ram_map.POKECENTER_MAP_IDS` from pret/pokered
constants (Viridian + Pewter + 8 later-game centers).

**Design rationale.**

1. **Kill the blackout exploit, hard.** Faint penalty bumped from -2 to
   -250 — chosen to dominate the ~+200-300 per-life exploration gain seen
   in V0.2.2 logs. Per-faint (not per-blackout): direct constant swap.
2. **Make PCs the *positive* heal route.** +5 for finding each new PC
   plus +2 for an actual heal-when-low gives the agent a clear reason
   to walk into a Pokemon Center voluntarily, where V0.2.2 gave none.
3. **Stop heal-spam.** Healing at near-full HP costs -1. Combined with
   step-penalty walking time, looping into a PC for a top-up is
   negative-EV.
4. **Don't refund blackouts.** When the game force-heals after whiteout,
   the heal-event detection would otherwise refund +2 and +5. Explicit
   `_blackout_pending` flag set when LOSE_BATTLE fires; suppresses next
   heal/visit reward and clears.

**Reward math at the new weights:**
- Faint a Pokemon in battle: **-250**
- Voluntary heal at new PC, low HP: **+5 + +2 = +7**
- Voluntary heal at known PC, low HP: **+2**
- Voluntary heal at any PC, near-full: **-1**
- Blackout into a PC: **-250** (the heal/visit refund is suppressed)
- First time entering Pewter PC: **+5 + +5 (NEW_MAP) + +0.3 (EXPLORE) = +10.3**

**Hypotheses to test in ELSA training:**
- Does the blackout loop die? (Strong prior: yes — -250 > +200-300.)
- Does the agent learn to visit PCs voluntarily? (Open — depends on
  whether PPO can credit-assign the +2/+5 sparse signal across the
  multi-step walk-to-PC sequence.)
- Does the -250 cause extreme risk aversion? (Possible failure mode:
  agent avoids battles entirely. Watch entropy + battle-count metrics.)
- Does the agent reach Pewter? (V1 milestone: clear Brock.)

**Status:** Config + sbatch ready
(`configs/elsa_brock_v0_2_3.yaml`, `slurm/train_brock_v0_2_3.sbatch`).
Launch on ELSA after V0.2.2 is cancelled (`scancel 14143`).

---

## V0.2.4 family — faint-magnitude sweep (after V0.2.3 paralysis)

**Files:** `RewardV0_2_4_f25`, `_f50`, `_f100` (each subclasses `RewardV0_2_3`
and only overrides `FAINT_PENALTY`).

**Why a sweep, not a single magnitude.** Watching V0.2.3 (job 14376
single-GPU + 14430 DDP, iter 200 each ≈ 1.6M steps each) showed the
agent never entered a single wild battle. The -250 faint penalty had
flipped the expected-value math:

  EV(fight) = P(win) × WIN_REWARD + (1 − P(win)) × (LOSE + FAINT)
            = P(win) × 10 + (1 − P(win)) × (−7 + −250)
            = 267 × P(win) − 257

For EV(fight) > EV(flee) = −1, you need **P(win) > 96.3%**. Random
init is nowhere near that. So the policy converges on "flee, accumulate
exploration reward, never engage." Death spiral: no fights → no battle
data → policy stays bad at fighting → fights stay scary → still no
fights.

This was flagged in the V0.2.3 design as a possible failure mode but
shipped anyway because the upside (kill the blackout exploit) was
clear. Lesson confirmed empirically: it dominates.

The V0.2.4 sweep tests **three magnitudes simultaneously** to find the
smallest faint penalty that still dominates per-life exploration gain
(~+200) without crossing the threshold into death-spiral territory.

| Variant | FAINT_PENALTY | EV-breakeven win% | Predicted behavior |
|---|---|---|---|
| f25  | -25  | (32 / 42) = 76% | Engages — uncertainty OK if survival likely |
| f50  | -50  | (57 / 67) = 85% | Marginal — high confidence required |
| f100 | -100 | (107 / 117) = 91% | Likely still paralyzes |

Inherits all other V0.2.3 logic (PC heal, blackout guard, new-map,
catch, trainer-win, etc.). Only the one number changes.

**ELSA runs.** Jobs 14455 (f25), 14456 (f50), 14457 (f100). All three
launched in parallel as DDP 4×L40S using V0.2.3's DDP scaffold —
first real use of the multi-GPU infrastructure. Each run gets one
full L40S node.

**Observed behavior at ~iter 1000-1160 (8-9M steps each):**

| | f25 | f50 | f100 |
|---|---|---|---|
| mean_return | +373 | +373 | +424 → +395 |
| episodes | 2304 | 2112 | 2112 |
| entropy | 0.3–0.7 oscillating | 0.30 flat | 0.18–0.63 (recent spike) |
| Fights in watch | **Yes, occasionally** | No | No |
| Enters buildings | No | No | No |

**Diagnosis.**
1. **f25 broke the death spiral on combat.** Confirmed by watching
   `iter_001100.pt`: agent does engage in wild encounters and sometimes
   wins. Entropy oscillation (0.3–0.7) confirms the policy is still
   considering alternatives instead of committing.
2. **f50 and f100 stayed paralyzed.** Same +373 return as f25 but
   from pure flee-and-explore, not combat. Entropy locked at 0.30
   (f50) — fully deterministic flee policy. f100's higher +424 is
   "most polished avoidance" (bigger stick → especially good at
   staying alive without engaging).
3. **A new failure mode surfaced even in f25**: the agent that
   *does* fight still **refuses to enter buildings or interact with
   NPCs**. Watching f25 iter_1100 showed: walks Route 1 grass, fights
   wild encounters, but stops dead at every doorway. Never enters
   Viridian PC / Mart / Forest. Reason inferred: policy learned
   A-press in the overworld is wasted ticks. Movement reliably pays
   tile exploration; A on empty tiles is a no-op. Step penalty
   (-0.001) makes every "wasted" A a small loss. So gradient prunes
   A out of the overworld action distribution. The policy presses
   A *only during battles* (context-conditional A, learned from
   battle-menu reinforcement).

**Lessons recorded.**
- Reward magnitude isn't just "make the bad thing bad." It shifts
  the *threshold* policies must clear before they'll attempt the
  action. -250 was correct for killing the blackout exploit but it
  also lifted the fight-EV threshold above the random-init policy's
  reach. The right magnitude has to be small enough that uncertain
  attempts are still positive-EV at moderate confidence.
- Parallel sweeps beat sequential single-magnitude runs for catching
  this kind of regime change. Cost was 3× compute (3 DDP runs in
  parallel), saved at least 2× wall-clock (we'd have iterated
  -25 → -50 → -100 sequentially otherwise).
- A second failure mode hides behind the first. Watching f25 only
  helped because the engagement-paralysis was solved. Until that
  unlocked, the building-paralysis was invisible (the agent that
  wouldn't fight also wouldn't walk into a doorway, but we'd have
  blamed faint penalty for everything).

**Status:** f25 continues running into 2026-06-04 morning to see how
far behavior develops; f50 and f100 cancelled at iter ~1060/1160 once
diagnosed. Documented separately in the late-night session note.

---

## V0.2.5 family — aggressive building / Mart / PC bonuses + entropy bump

**Files:** `RewardV0_2_5` (base, subclasses `RewardV0_2_4_f25`), and
`RewardV0_2_5_b20`, `_b50`, `_b100` (each overrides `NEW_MAP_REWARD`).

**Why this design.** V0.2.4_f25 unlocked battles but exposed the
overworld-A-is-wasted-ticks problem: agent won't enter buildings,
talk to NPCs, or interact with anything that requires A in the
overworld. For V1 (Brock), the agent MUST:
1. Walk into Brock's Gym (a building map_id transition)
2. Press A on Brock to trigger the gym battle
3. Win the battle (already-learned skill from V0.2.4_f25 evidence)

(1) is a movement action, not an A-press, so a strong enough reward
for "entered a new building map_id" should solve it. (2) needs the
A-press-in-overworld habit, which requires a different intervention —
deferred to V0.2.6.

**Changes from V0.2.4_f25:**

| Signal | V0.2.4_f25 | V0.2.5 |
|---|---|---|
| NEW_MAP_REWARD | +5 | **swept: +20 / +50 / +100** |
| Mart or PC first-visit bonus | (none) | **+20 stacked** on NEW_MAP |
| PC_HEAL_LOW | +2 | **+5** (heal becomes more attractive) |
| `entropy_coef` (PPO config) | 0.01 | **0.1** (10× regularization) |

`MART_PC_BONUS` fires once per (map_id, first-visit) when the new
map's ID is in `POKECENTER_MAP_IDS ∪ POKEMART_MAP_IDS`. Brock's Gym
is NOT a Mart/PC (its map_id is 0x36), so it only collects the base
NEW_MAP_REWARD — adding gym map_ids to the bonus list is a clear
V0.2.6 candidate once V0.2.5's effect is measured.

`POKEMART_MAP_IDS` added to `ram_map.py`: 8 Pokemarts from
pret/pokered (Viridian + Pewter are V1-critical; Celadon Dept Store
sub-floors deliberately excluded — they register as their own new
map_ids and only collect the base reward).

**Reward math at the new weights** (b50 used as example):

| Action | Reward total |
|---|---|
| Enter Viridian City (outdoor) | +50 NEW_MAP + +0.3 EXPLORE = +50.3 |
| Enter Viridian Mart, first time | +50 + +20 MART_PC + +0.3 = +70.3 |
| Enter Viridian PC, first time, at low HP | +50 + +20 + +5 PC_FIRST + +5 PC_HEAL_LOW = +80.3 |
| Enter Brock's Gym, first time | +50 NEW_MAP + +0.3 = +50.3 |
| Beat a wild Pidgey | +10 BEAT_MON (unchanged) |
| Faint a Pokemon | -25 FAINT_PENALTY (inherited V0.2.4_f25) |

**Why entropy_coef 0.1.** The V0.2.4 sweep showed entropy collapsing
from 1.94 → 0.30 over ~1000 iters with the PPO default `entropy_coef
= 0.01`. f50's entropy locked at 0.30 = fully deterministic
flee-policy. Even f25's entropy was oscillating 0.3–0.7, with the
0.3 floor suggesting partial commitment. With overnight training
budget (~5 hours sleep on 48h SLURM jobs ≈ thousands more iters),
the V0.2.5 runs would otherwise risk the same collapse before any
building-reward signal had time to take effect. 10× the
regularization keeps the policy plastic long enough for new building
visits (which are rare-by-construction) to actually accumulate the
new gradient signal.

**Risk acknowledged:** if 0.1 is too high, the policy never commits
and we see flat returns + flat 1.9+ entropy across all three b
variants in the morning. That's a recoverable failure mode (drop to
0.05 next run); the asymmetric risk favors over-regularization
overnight when we can't intervene.

**Why sweep NEW_MAP_REWARD and not Mart/PC bonus or PC_HEAL.** The
single biggest unknown is how big a building-entry reward needs to
be to overcome the policy's current "stay in grass, explore tiles
that pay +0.3" attractor. A full 4096-step episode in grass caps
around +100 from raw EXPLORE. So:
- +20 might not be enough to outweigh a worse episode in a building.
- +50 should clearly dominate a single tile's exploration.
- +100 is overkill — included to bracket the unknown and check
  whether the agent over-optimizes for building entries (e.g.
  refuses to fight on routes because building entries pay so much).

Mart/PC bonus is held at +20 as a fixed "small extra for the things
that matter for survival." PC_HEAL_LOW bumped to +5 to scale roughly
with the new building rewards (V0.2.3's +2 was tuned against the
old +5 NEW_MAP — keeping the same ratio).

**ELSA runs.** Jobs 14511 (b20), 14512 (b50), 14513 (b100). All
three launched in parallel as DDP 4×L40S alongside the still-running
V0.2.4_f25 14455 (kept running as a control to see how far the
V0.2.4_f25 policy develops with more time).

**Hypotheses to test in the morning:**
1. Does any V0.2.5 variant enter buildings? (Walk into Viridian PC
   / Mart at any point would be visible in mean_return jumps.)
2. Does entropy_coef = 0.1 actually keep entropy high, or does
   the policy commit to a different deterministic strategy anyway?
3. b20 vs b50 vs b100: is there a meaningful behavioral difference,
   or do they all hit the same building-entry threshold?
4. Does Brock's Gym entry ever happen? (Reward signal is only
   +NEW_MAP_REWARD without the Mart/PC bonus — may need a gym list
   in V0.2.6.)

**ELSA runs.** Jobs 14511 (b20), 14512 (b50), 14513 (b100). DDP 4×L40S
each. Ran ~8 hours overnight, all cancelled morning 2026-06-04 once
results clear.

**Observed at iter ~16-17k (~130-140M samples each):**

| | b20 | b50 | b100 |
|---|---|---|---|
| mean_return | +504 | +686 | **+796** |
| episodes | 34144 | 31808 | 32032 |
| tiles | 15076 | 14879 | **1841** |
| entropy | 1.82 | 1.84 | **1.87** |

**Diagnosis.** Two layers of regression vs V0.2.4_f25:

1. **`entropy_coef = 0.1` was too high.** All three V0.2.5 entropies
   are at ~1.85, essentially uniform (max is ln(7) = 1.95). The
   policy did not commit to anything useful. Strong rewards on
   specific contexts (battle-flee, doorway-entry) did get learned —
   PPO entropy regularization averages across states, so individual
   high-gradient contexts can still commit while the overall policy
   stays exploratory — but the broad strategy is random walking.
   - The asymmetric-risk gamble lost on the "too high" side.
   - Watching b20 iter_17900 / b50 iter_16500 confirmed visually:
     overworld movement is near-random, but flee-in-battle and
     enter-doorway are consistent.

2. **NEW_MAP_REWARD scaling broke f25's combat balance.** Even at
   b20's +20 (4× f25's +5), the EV math flipped: flee-to-explore
   beats fight-for-XP. New observation in watch: V0.2.5 agents
   **flee every wild battle** — they correctly optimized the new
   reward function we gave them. V0.2.4_f25, with NEW_MAP at +5,
   had organic strategic combat (fled only Pidgey for type
   disadvantage, fought everything else). The +20 magnitude alone
   was enough to lift the optimal strategy off the combat path.

**Surprising positives in V0.2.5:**
- b20 agent **talked to an NPC and received a free potion**. First
  verified NPC interaction in any V0.x. The high entropy was doing
  exploration work — random A presses occasionally hit NPCs and the
  policy didn't unlearn them because there's no penalty. Validates
  that V0.2.7's dialog reward (when ready) has a behavioral baseline
  to amplify.
- All three V0.2.5 agents **entered buildings on purpose** (as a
  reward-targeting behavior, not just incidental). The building-bonus
  signal worked.

**Lesson recorded.**
- **Bonus magnitudes must respect the existing balance.** Adding a
  large reward for one behavior implicitly de-prioritizes everything
  else by EV ratio. Always design bonuses as small additions on
  top of a working balance, not as dominant terms that reshape the
  optimum.
- **PPO entropy is a global average.** Bumping `entropy_coef`
  doesn't enforce per-context randomness — strong-gradient contexts
  still commit. This is useful for "keep exploring weakly-rewarded
  states" but doesn't prevent the policy from converging on a bad
  global strategy if the rewards point there.
- **Combat is the V1 critical path; never break it for cosmetic
  rewards.** V0.2.5 broke combat for building entries. Both are
  needed; design the next iteration to preserve combat behavior
  while ADDING building/gym signals.

**Status:** All three cancelled 2026-06-04 morning.

---

## V0.2.6 — restore f25 balance + small Mart/PC + Gym bonuses

**File:** `RewardV0_2_6` (subclasses `RewardV0_2_4_f25` — the version
whose combat balance is proven to work).

**Why this design.** V0.2.5 taught us that the f25 reward structure
produces useful battle behavior and *must not be regressed*. V0.2.5
also taught us that small targeted bonuses can teach building entry
without reshaping the global optimum — as long as the bonuses stay
small relative to combat rewards.

**Changes from V0.2.4_f25:**

| Signal | V0.2.4_f25 | V0.2.6 |
|---|---|---|
| MART_PC_BONUS | (none) | **+10** stacked on NEW_MAP for first visit to a PC or Mart |
| GYM_BONUS | (none) | **+20** stacked on NEW_MAP for first visit to a gym map_id |
| `entropy_coef` | 0.01 | **0.03** (3× — between f25's collapse-prone 0.01 and V0.2.5's commit-prevention 0.1) |

Everything else is V0.2.4_f25 unchanged: FAINT -25, NEW_MAP +5,
BEAT_MON +10, FLEE_PENALTY -1, full V0.2.3 PC heal logic + blackout
guard. PC_HEAL_LOW is back to +2 (V0.2.5 bumped to +5; V0.2.6 restores).

**Reward math comparison** (key actions, V0.2.4_f25 vs V0.2.6):

| Action | V0.2.4_f25 | V0.2.6 |
|---|---|---|
| New tile in grass | +0.3 | +0.3 |
| Beat wild Pidgey | +10 | +10 |
| Enter Viridian PC, first time, low HP | +5+5+2 = +12 | +5+5+2+10 = **+22** |
| Enter Viridian Mart, first time | +5 | +5+10 = **+15** |
| **Enter Pewter Gym (Brock), first time** | +5 | +5+20 = **+25** |
| 3-mon trainer battle win | +50 | +50 |
| Faint a Pokemon | -25 | -25 |

The largest building bonus (Pewter Gym entry, +25) is half of a
full trainer battle win (+50) and equal to two wild Pidgey kills.
Combat stays the dominant reward source — building bonuses are
"destination markers," not goals.

`GYM_MAP_IDS` initially contains ONLY `MAP_PEWTER_GYM = 0x36`
(verified for V1; the other 7 gym map_ids from memory had a
collision with the Cerulean Mart 0x41 — collisions surfaced by
the test_v26_gym_and_mart_lists_disjoint assertion. The other
gyms need empirical verification on the ROM before being added).

**Why entropy_coef = 0.03 specifically.** f25 at iter 18900 (155M
steps) had entropy down to 0.29 — strongly committed to a few
visual heuristics (avoid Pidgey at high HP, fight at 1 HP, enter
buildings opportunistically for tile rewards). That commitment is
good for fast convergence but bad for never exploring further
strategies (e.g. would f25 ever discover "talking to NPCs is
rewarding" with entropy at 0.29? Probably not within wall-clock
budgets). 0.03 should slow that collapse: with 3× regularization
we'd expect entropy to land around 0.5-0.8 at 100M+ samples,
preserving some plasticity without the "everything is random"
problem V0.2.5 had at 0.1.

**Dialog detection (V0.2.7 candidate).** Still deferred. The
empirical observation from V0.2.5 b20 — agent talked to NPC,
got potion, learned nothing because no reward signal fired —
is exactly the gap dialog reward would fill. Once we verify the
RAM byte for "text box active" (likely in the 0xCF80-0xCFD0 range
based on pret/pokered, but unverified), V0.2.7 = V0.2.6 + dialog
reward per unique (map_id, x, y) where dialog opens.

**Hypotheses to test:**
1. Does V0.2.6 reach Pewter Gym? (The +25 GYM_BONUS should pull
   the policy toward Pewter once an episode finds Route 2 → Forest →
   Pewter. f25 reached Viridian buildings by 155M; V0.2.6 should
   reach further given the explicit destination reward.)
2. Does V0.2.6 keep f25's combat behavior? (Watch checkpoint for
   strategic fleeing vs flee-everything.)
3. Does entropy land in the 0.5-0.8 range as predicted? (If it
   collapses below 0.3, we may need to go to 0.05 in a follow-up.)
4. Does the agent press A on Brock once it enters the gym? (The
   existing FLAG_REWARD +10 fires when the Brock-challenged story
   flag sets — this is sparse but might be enough given the high
   entropy. If not, V0.2.7 dialog reward is the fix.)

**ELSA run.** Job 15929. DDP 4×L40S on gpu-node019. Ran ~1 hour
(~19M steps, iter 2302) before being cancelled in favor of V0.2.7.

**Observed at iter 2302:**

| | V0.2.6 |
|---|---|
| mean_return | +370 |
| episodes | 4576 |
| tiles | 26108 |
| entropy | **0.75** |

The entropy landing at 0.75 is exactly the predicted 0.5-0.8 sweet
spot — `entropy_coef = 0.03` worked structurally as intended. So
the V0.2.5 "too-random" failure mode is solved by going to 0.03.

**Diagnosis.** Watching iter_002300 showed the policy is **still
fleeing every wild battle**. The +10/+20 Mart/PC/Gym bonuses
weren't large enough to swing combat EV — they shouldn't have been,
because they only fire on first-visit per location, which is rare.
The actual problem is upstream: V0.2.6 inherits V0.2.4_f25's
`BEAT_MON = +10`, which sets the fight-EV breakeven at **74% win
confidence** (`P × 10 + (1-P) × (-7 + -25) > -1 → P > 31/42`). A
partially-trained policy isn't anywhere near 74% on most encounters.

Caveat: f25 itself didn't show strategic combat until ~155M steps.
V0.2.6 at 19M is genuinely too early to declare it broken. But:
- The breakeven math is structural and won't improve with more time
- Iterating the design is cheaper than waiting 8× longer to confirm
- V0.2.7's combat bump is independent improvement either way

**Lesson recorded.**
- **f25's +10 BEAT_MON was tuned for a fully-trained policy.** It
  forms a stable equilibrium at high win confidence but provides
  no gradient toward fighting from a low-confidence starting point.
  Early-policy combat needs lower breakeven thresholds.
- **First-visit bonuses are too sparse to affect overall strategy.**
  Mart/PC/Gym +10/+20 stays in the noise compared to per-fight EV.
- **Watch + iterate beats wait + hope.** Cheaper to cancel at 19M
  and ship a better hypothesis than burn 130M more samples on a
  policy that the math says won't converge to combat.

**Status:** Cancelled mid-morning 2026-06-04, replaced by V0.2.7.

---

## V0.2.7 — combat reward bump (lower the fight-EV threshold)

**File:** `RewardV0_2_7` (subclasses `RewardV0_2_6`).

**Why this design.** V0.2.6 still produced flee-everything because
the underlying f25 `BEAT_MON = +10` makes uncertain battles
negative-EV. The fix is direct: bump combat rewards so the fight
threshold is reachable from a partially-trained policy.

**Changes from V0.2.6:**

| Signal | V0.2.6 | V0.2.7 | Breakeven win % |
|---|---|---|---|
| BEAT_MON_REWARD | +10 | **+20** | 74% → 60% |
| TRAINER_WIN_BONUS | +20 | **+30** | (Route 2 trainers are V1 critical path) |

Everything else inherited unchanged: FAINT -25, NEW_MAP +5,
MART_PC_BONUS +10, GYM_BONUS +20, PC heal logic, blackout guard,
`entropy_coef` 0.03.

**EV math at the new BEAT_MON.** With BEAT_MON +20, FAINT -25,
LOSE -7:

  EV(fight) = P × 20 + (1−P) × (−32) = 52P − 32
  EV(fight) > EV(flee = −1):  P > 31/52 = 59.6%

60% win confidence is reachable from a few wild-battle samples
where the agent has Bulbasaur grass-attacks at level 5-6 (super
effective against the common Pidgey/Rattata/Caterpie distribution
on Route 1). The agent doesn't need to be good — just slightly
better than coin-flip.

**Reward math comparison** (cumulative through V0.2.7):

| Action | V0.2.4_f25 | V0.2.6 | V0.2.7 |
|---|---|---|---|
| Beat wild Pidgey | +10 | +10 | **+20** |
| 3-mon trainer battle win | +50 | +50 | **+90** |
| Faint a Pokemon | -25 | -25 | -25 |
| Enter Pewter Gym, first time | +5 | +25 | +25 |

A 3-mon trainer win (+90) is now equivalent to **3.6× a Pewter Gym
discovery**. Combat is the dominant reward source again, which is
the goal: the agent should be playing Pokemon, not playing
geography.

**Naming clarification.** V0.2.7 was originally penciled for
dialog-detection reward. That's now V0.2.8, still pending
RAM-byte verification.

**Status:** Single DDP run launched 2026-06-04 ~12:00 PM as job
16185 on gpu-node019. Cooking.

**Hypotheses to test:**
1. Does the agent fight more wild battles? (Watch action
   distribution: FIGHT vs RUN ratio in battle menus.)
2. Does the agent pursue trainers? (TRAINER_WIN_BONUS bump should
   pull policy toward visible-trainer maps like Route 2.)
3. Does the agent reach Pewter / Brock's Gym? (Combat-grind path
   should now align with the geography reward signals.)
4. Does V0.2.7 preserve the entropy ~0.75 we saw in V0.2.6?

---

## Cross-cutting design patterns learned (writeup source)

These are the meta-lessons that should make it into the eventual
portfolio writeup, beyond per-version detail:

1. **Reward loopholes are inevitable; iteration is the protocol.**
   Every V0.2.x has shipped with an exploit or failure mode that
   wasn't visible in the design but was obvious from watching the
   trained policy. Sequence: V0.2.1 flee-as-win, V0.2.2
   blackout-as-free-heal, V0.2.3 fight-paralysis, V0.2.4_f25
   A-press-is-wasted, V0.2.5 flee-to-explore, V0.2.6 still-flees.
   **Plan to iterate; don't try to design a complete reward
   upfront.**

2. **EV math at the random-init confidence level is the right
   sanity check.** Most failure modes in this project were
   predictable from the EV table BEFORE training started:
   - V0.2.3 -250 faint → fight breakeven at 96% confidence
     (impossible for random init)
   - V0.2.4_f25 +10 BEAT_MON → 74% (high; needed 155M steps)
   - V0.2.7 +20 BEAT_MON → 60% (reachable with type advantage)
   Running this math during design surfaces structural problems
   before compute is committed.

3. **Bonus magnitudes implicitly re-rank ALL behaviors.** Adding
   a +20 reward for behavior X means every other behavior is now
   ~+20 less attractive in opportunity cost. V0.2.5's lesson:
   even a +20 NEW_MAP bonus was enough to make flee-to-explore
   beat fight-for-XP. Always design bonuses as small additions
   on top of a working balance, not as dominant terms.

4. **PPO entropy regularization is a global average, not a
   per-context floor.** Increasing `entropy_coef` keeps
   weakly-rewarded states stochastic but doesn't prevent
   strong-gradient contexts from committing. V0.2.5's lesson:
   the policy fled all battles even with entropy_coef=0.1
   because the flee reward gradient was strong; only overworld
   movement stayed random.

5. **Watching > metrics for diagnostic.** Almost every diagnosis
   in this project came from watching the trained policy in the
   SDL2 window, not from looking at mean_return / entropy / etc.
   Returns rose monotonically in every variant; only watching
   revealed which variants were actually learning the V1 critical
   path. **The metrics CSV is the trail; the watch is the
   ground truth.**

6. **Parallel sweeps beat sequential single-magnitude runs.** The
   V0.2.4 (f25/f50/f100) and V0.2.5 (b20/b50/b100) sweeps caught
   regime changes (where the optimal strategy flips) in 1× the
   wall-clock that sequential would have taken. Cost: 3× the
   compute. Worth it for any question shaped "which magnitude
   range crosses the threshold."

7. **Multi-agent parallel review pre-implementation caught 8+
   bugs in the DDP code BEFORE writing.** The reviewer subagent
   pattern (2 parallel researchers reviewing the plan from
   different angles, third agent code-reviewing the implementation)
   produced concrete fixes that would have been silent failures
   otherwise: AsyncVectorEnv context="spawn" requirement, OMP env
   vars before torch import, modern NCCL var name, single
   srun--ntasks=1 to avoid the double-launch hang, NCCL warmup,
   teardown order. Adopt this pattern for any non-trivial new
   infrastructure.

---

## Tracker for future versions

| Version | Status | Headline change | Outcome |
|---|---|---|---|
| V0.1 | Cancelled | First attempt | Confounded by env button-hold bug |
| V0.2 | Cancelled | + movement bonus + battle suite | Entropy collapse to 0 by iter 1100 |
| V0.2.1 | Cancelled | exploration scaled 3× down | Avoided collapse but exposed flee-as-win exploit |
| V0.2.2 | Cancelled | flee fix + new-map / catch rewards | Healthy metrics but blackout-as-free-heal exploit |
| V0.2.3 | Cancelled | PC heal rewards + -250 faint + blackout guard | Killed blackout but agent refused to fight (paralysis) |
| V0.2.4_f25 | Cancelled | -25 faint (relaxed) | Engages strategically (155M steps), enters buildings for tiles, no NPC interaction |
| V0.2.4_f50 | Cancelled | -50 faint | Flee-and-explore, entropy collapsed |
| V0.2.4_f100 | Cancelled | -100 faint | Highest return from "polished avoidance," no fights |
| V0.2.5_b20 | Cancelled | +20 NEW_MAP + Mart/PC stack + entropy_coef 0.1 | Random-walks, flees all fights, DID interact with NPC accidentally (no reward fired) |
| V0.2.5_b50 | Cancelled | +50 NEW_MAP + Mart/PC stack | Same as b20, slightly higher returns from map jackpots |
| V0.2.5_b100 | Cancelled | +100 NEW_MAP + Mart/PC stack | "Polished avoidance," only 1841 tiles, agent stuck in few rooms |
| V0.2.6 | Cancelled | f25 balance + Mart/PC +10 + Gym +20 + entropy 0.03 | entropy landed at 0.75 (perfect); still flees all wild battles (BEAT_MON +10 EV breakeven 74%) |
| V0.2.7 | Running | + BEAT_MON +20 + TRAINER_WIN +30 | TBD (job 16185) |
| V0.2.8 (planned) | — | + dialog detection reward per unique (map, x, y) | Pending RAM verification |
| V1.0 | Reserved | First version to clear Brock | — |
