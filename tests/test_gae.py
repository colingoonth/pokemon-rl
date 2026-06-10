"""Correctness tests for GAE + the truncation-bootstrap fix.

These lock in the two CRITICAL fixes from the day-8 audit:
  - truncation is bootstrapped, not treated as a hard terminal
  - GAE masks correctly at an episode boundary (no cross-episode value leak)

Expected values are hand-computed in the comments so a future change that
shifts the numbers is caught and has to justify itself.
"""
from __future__ import annotations

import numpy as np
import torch

from pokerl.agent.ppo import RolloutBuffer, apply_truncation_bootstrap


def _buf_from(rewards, values, dones):
    """Build a (n_steps, 1-env) RolloutBuffer with given per-step tensors."""
    n = len(rewards)
    buf = RolloutBuffer(
        obs=torch.zeros((n, 1, 1), dtype=torch.uint8),
        actions=torch.zeros((n, 1), dtype=torch.long),
        log_probs=torch.zeros((n, 1)),
        values=torch.tensor(values, dtype=torch.float32).reshape(n, 1),
        rewards=torch.tensor(rewards, dtype=torch.float32).reshape(n, 1),
        dones=torch.tensor(dones, dtype=torch.float32).reshape(n, 1),
    )
    return buf


def test_gae_no_boundary_matches_hand_computation():
    """Plain 3-step GAE with no episode boundary, hand-checked.

    gamma=0.99 lambda=0.95, r=[1,2,3] V=[.5,.6,.7] lastV=.8 dones=0 last_done=0
      t2: d=3+.99*.8-.7=3.092            gae2=3.092
      t1: d=2+.99*.7-.6=2.093            gae1=2.093+.9405*3.092=5.00105
      t0: d=1+.99*.6-.5=1.094            gae0=1.094+.9405*5.00105=5.79749
    returns = adv + V
    """
    buf = _buf_from([1, 2, 3], [0.5, 0.6, 0.7], [0, 0, 0])
    buf.compute_gae(
        last_values=torch.tensor([0.8]),
        last_dones=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    adv = buf.advantages.reshape(-1).tolist()
    assert adv[2] == __import__("pytest").approx(3.092, abs=1e-4)
    assert adv[1] == __import__("pytest").approx(5.00105, abs=1e-4)
    assert adv[0] == __import__("pytest").approx(5.79749, abs=1e-4)
    ret = buf.returns.reshape(-1).tolist()
    assert ret == __import__("pytest").approx([6.29749, 5.60105, 3.792], abs=1e-4)


def test_gae_masks_at_boundary_no_cross_episode_leak():
    """A done at dones[2] cuts the bootstrap + gae propagation at t=1.

    dones=[0,0,1] means step t=1's transition ended the episode, so at t=1:
      next_nonterminal = 1 - dones[2] = 0
      t2: d=3+.99*.8-.7=3.092            gae2=3.092   (own episode's tail)
      t1: d=2+.99*.7*0-.6=1.4            gae1=1.4     (NO leak from gae2)
      t0: d=1+.99*.6-.5=1.094            gae0=1.094+.9405*1.4=2.41070
    The key assertion: adv[1]==1.4 exactly — t=2's value does NOT bleed back
    across the boundary.
    """
    buf = _buf_from([1, 2, 3], [0.5, 0.6, 0.7], [0, 0, 1])
    buf.compute_gae(
        last_values=torch.tensor([0.8]),
        last_dones=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    adv = buf.advantages.reshape(-1).tolist()
    assert adv[1] == __import__("pytest").approx(1.4, abs=1e-4)
    assert adv[2] == __import__("pytest").approx(3.092, abs=1e-4)
    assert adv[0] == __import__("pytest").approx(2.41070, abs=1e-4)


def test_truncation_bootstrap_only_truncated_envs():
    """gamma*V(final_obs) is added to TRUNCATED envs, not normal/terminated ones.

    3 envs: env0 normal, env1 truncated, env2 terminated.
    V(final_obs)=10 for any obs, gamma=0.9, rewards=[1,2,3].
      env0: no final_obs   -> 1 (unchanged)
      env1: trunc, ~term   -> 2 + 0.9*10 = 11
      env2: term           -> 3 (unchanged: a true terminal has no future)
    """
    term_np = np.array([False, False, True])
    trunc_np = np.array([False, True, True])  # env2 both flagged; ~term excludes it
    final = np.empty(3, dtype=object)
    final[1] = np.array([5, 5], dtype=np.uint8)
    final[2] = np.array([6, 6], dtype=np.uint8)
    info = {"_final_obs": np.array([False, True, True]), "final_obs": final}

    rewards = torch.tensor([1.0, 2.0, 3.0])
    out = apply_truncation_bootstrap(
        rewards, term_np, trunc_np, info,
        value_fn=lambda o: torch.full((o.shape[0],), 10.0),
        gamma=0.9, n_envs=3, device="cpu",
    )
    assert out.tolist() == [1.0, 11.0, 3.0]
    # input not mutated
    assert rewards.tolist() == [1.0, 2.0, 3.0]


def test_truncation_bootstrap_noop_when_nothing_truncates():
    info = {"_final_obs": np.array([False, False])}
    rewards = torch.tensor([1.0, 2.0])
    out = apply_truncation_bootstrap(
        rewards, np.array([False, False]), np.array([False, False]), info,
        value_fn=lambda o: torch.zeros(o.shape[0]),
        gamma=0.99, n_envs=2, device="cpu",
    )
    assert out.tolist() == [1.0, 2.0]
