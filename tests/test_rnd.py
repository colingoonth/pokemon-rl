"""Tests for the RND curiosity module + the two-head warm-start path."""
from __future__ import annotations

import numpy as np
import torch

from pokerl.agent.networks import ActorCritic
from pokerl.agent.rnd import RewardForwardFilter, RNDModel, RunningMeanStd

FRAME = (1, 72, 80)
PROG = 4


def test_running_mean_std_matches_numpy():
    rms = RunningMeanStd(shape=(3,))
    data = np.random.RandomState(0).randn(500, 3) * 2.0 + 1.0
    # feed in two batches to exercise the parallel update
    rms.update(data[:300])
    rms.update(data[300:])
    assert np.allclose(rms.mean, data.mean(0), atol=1e-6)
    assert np.allclose(rms.var, data.var(0), atol=1e-4)


def test_reward_forward_filter_discounts():
    rff = RewardForwardFilter(gamma=0.9, n_envs=2)
    r1 = rff.update(np.array([1.0, 0.0]))
    r2 = rff.update(np.array([1.0, 0.0]))
    # second return = r + gamma * prev = 1 + 0.9*1 = 1.9 for env0
    assert np.allclose(r1, [1.0, 0.0])
    assert np.allclose(r2, [1.9, 0.0])


def test_target_frozen_predictor_trainable():
    m = RNDModel(FRAME, PROG)
    assert all(not p.requires_grad for p in m.target.parameters())
    assert all(p.requires_grad for p in m.predictor.parameters())


def test_novelty_is_nonneg_and_shaped():
    m = RNDModel(FRAME, PROG)
    frame = torch.randn(5, *FRAME)
    prog = torch.zeros(5, PROG)
    nov = m.novelty(frame, prog)
    assert nov.shape == (5,)
    assert (nov >= 0).all()


def test_predictor_training_reduces_novelty():
    """Training the predictor on a fixed batch should drop its novelty —
    the core RND mechanism (seen states become predictable -> boring)."""
    torch.manual_seed(0)
    m = RNDModel(FRAME, PROG)
    frame = torch.randn(8, *FRAME)
    prog = torch.randint(0, 2, (8, PROG)).float()
    opt = torch.optim.Adam(m.predictor.parameters(), lr=1e-3)
    before = m.novelty(frame, prog).mean().item()
    for _ in range(50):
        loss = m.novelty(frame, prog).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = m.novelty(frame, prog).mean().item()
    assert after < before * 0.5, f"novelty did not drop: {before:.4f} -> {after:.4f}"


def test_target_unchanged_by_predictor_training():
    """Training must not move the frozen target."""
    torch.manual_seed(0)
    m = RNDModel(FRAME, PROG)
    frame = torch.randn(4, *FRAME)
    prog = torch.zeros(4, PROG)
    t_before = m.target(frame, prog).detach().clone()
    opt = torch.optim.Adam(m.predictor.parameters(), lr=1e-2)
    for _ in range(10):
        loss = m.novelty(frame, prog).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    t_after = m.target(frame, prog).detach()
    assert torch.allclose(t_before, t_after)


def _ram_model():
    return RNDModel(FRAME, PROG, input_mode="ram", num_maps=256,
                    n_fourier=128, fourier_scale=16.0)


def test_ram_mode_shapes_and_freeze():
    """RAM-mode RND: target frozen, predictor trainable, novelty (B,) non-neg."""
    m = _ram_model()
    assert all(not p.requires_grad for p in m.target.parameters())
    assert all(p.requires_grad for p in m.predictor.parameters())
    state = torch.tensor([[40., 5., 7.], [12., 3., 9.]])
    nov = m.novelty(state, torch.zeros(2, PROG))
    assert nov.shape == (2,)
    assert (nov >= 0).all()


def test_ram_novelty_decays_but_neighbour_stays_novel():
    """Revisiting a tile drives its novelty toward zero (loiter self-limits),
    while an UNVISITED adjacent tile stays novel — the per-tile resolution that
    raw-float coords would smear away. This is the Day-11 fix for the flat
    frame-RND novelty + overworld-animation loiter trap."""
    torch.manual_seed(0)
    m = _ram_model()
    prog = torch.zeros(1, PROG)
    tile = torch.tensor([[40., 5., 7.]])
    opt = torch.optim.Adam(m.predictor.parameters(), lr=1e-3)
    before = m.novelty(tile, prog).item()
    for _ in range(200):
        loss = m.novelty(tile, prog).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = m.novelty(tile, prog).item()
    neighbour = m.novelty(torch.tensor([[40., 6., 7.]]), prog).item()
    assert after < before * 0.2, f"tile novelty did not decay: {before:.4f} -> {after:.4f}"
    assert neighbour > after * 5, f"neighbour over-smoothed: {neighbour:.4f} vs {after:.4f}"


def test_ram_progress_flag_refreshes_novelty():
    """Flipping a story bit on an over-visited tile re-spikes novelty — the
    mechanism that pulls the agent to backtrack (parcel -> pokedex -> pokeballs)
    across an already-explored map, which frame-RND could not provide."""
    torch.manual_seed(0)
    m = _ram_model()
    prog0 = torch.zeros(1, PROG)
    tile = torch.tensor([[40., 5., 7.]])
    opt = torch.optim.Adam(m.predictor.parameters(), lr=1e-3)
    for _ in range(200):
        loss = m.novelty(tile, prog0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    seen = m.novelty(tile, prog0).item()
    prog1 = torch.tensor([[1., 0., 0., 0.]])      # got_oaks_parcel flips
    refreshed = m.novelty(tile, prog1).item()
    assert refreshed > seen * 5, f"flag flip did not refresh novelty: {seen:.4f} -> {refreshed:.4f}"


def test_warmstart_two_head_checkpoint_loads_strict_false():
    """A pre-RND (no critic_int) checkpoint must warm-start into the 3-head
    net with critic_int as the ONLY missing key, and no unexpected keys."""
    new_net = ActorCritic((4, 72, 80), n_actions=7)
    old_state = {k: v for k, v in new_net.state_dict().items()
                 if not k.startswith("critic_int")}
    missing, unexpected = new_net.load_state_dict(old_state, strict=False)
    assert unexpected == []
    assert all(k.startswith("critic_int") for k in missing)
    assert len(missing) == 2  # weight + bias
