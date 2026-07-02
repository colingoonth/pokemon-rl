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


def test_step0_progress_head_padding_preserves_policy():
    """V0.5.9 warm-start surgery: a pre-Step-0 checkpoint (narrow, progress-blind
    heads) must load into the progress-fused net, start BEHAVIORALLY IDENTICAL
    with zero progress, and leave the new progress columns at zero."""
    from pokerl.agent.networks import ActorCritic as AC
    from pokerl.agent.ppo import pad_progress_head_weights

    # "Old" net: no progress fusion (progress_dim=0 -> heads take hidden only).
    old = AC((4, 72, 80), n_actions=7, progress_dim=0)
    old.eval()
    # "New" net: Step-0 net with the 4 progress bits fused at the head input.
    new = AC((4, 72, 80), n_actions=7, progress_dim=PROG)

    state = dict(old.state_dict())
    pad_progress_head_weights(state, new.state_dict())
    missing, unexpected = new.load_state_dict(state, strict=False)
    assert unexpected == []
    new.eval()

    # New progress columns are exactly zero after the pad.
    assert torch.count_nonzero(new.actor.weight[:, -PROG:]) == 0
    assert torch.count_nonzero(new.critic.weight[:, -PROG:]) == 0

    # With zero progress bits, the new net reproduces the old net's outputs.
    obs = torch.randint(0, 256, (5, 4, 72, 80), dtype=torch.uint8)
    zero_prog = torch.zeros(5, PROG)
    with torch.no_grad():
        old_logits, old_v, _ = old(obs, torch.zeros(5, 0))
        new_logits, new_v, _ = new(obs, zero_prog)
    assert torch.allclose(old_logits, new_logits, atol=1e-6)
    assert torch.allclose(old_v, new_v, atol=1e-6)
    # NB: because the progress columns start at ZERO, flipping the bits has no
    # effect on the freshly-padded net — that only emerges once training moves
    # those weights off zero. The zero-column check above is the invariant.


# --- V0.5.6 count-based episodic novelty ------------------------------------

def test_count_bonus_decays_with_revisits():
    """coef/sqrt(N+1): a fresh cell pays coef, revisits diminish monotonically."""
    from pokerl.agent.ppo import count_novelty_bonus
    d = {}
    cell = (0x0C, 5, 7, (0, 0, 0, 0))
    b1 = count_novelty_bonus(d, cell, 1.0)
    b2 = count_novelty_bonus(d, cell, 1.0)
    b3 = count_novelty_bonus(d, cell, 1.0)
    assert b1 == 1.0                       # N=0 -> 1/sqrt(1)
    assert np.isclose(b2, 1.0 / np.sqrt(2))
    assert np.isclose(b3, 1.0 / np.sqrt(3))
    assert b1 > b2 > b3                     # strictly decreasing
    assert 0.0 < b3 <= 1.0                  # bounded in (0, coef]


def test_count_bonus_progress_flip_reactivates_same_tile():
    """The standout mechanism: the SAME (map,x,y) tile, once its progress bits
    flip (parcel obtained), is a fresh cell again -> full bonus. This is what
    re-lights the already-walked Oak backtrack within an episode."""
    from pokerl.agent.ppo import count_novelty_bonus
    d = {}
    xy = (0x0C, 5, 7)
    before = xy + ((0, 0, 0, 0),)           # pre-parcel
    after = xy + ((1, 0, 0, 0),)            # got_oaks_parcel flipped
    # walk the tile down to a low bonus pre-parcel
    for _ in range(9):
        count_novelty_bonus(d, before, 1.0)
    stale = count_novelty_bonus(d, before, 1.0)   # 10th visit: 1/sqrt(10)
    refreshed = count_novelty_bonus(d, after, 1.0)  # same tile, new progress: 1/sqrt(1)
    assert np.isclose(stale, 1.0 / np.sqrt(10))
    assert refreshed == 1.0
    assert refreshed > stale * 3            # reactivation is a large jump


def test_count_bonus_distinct_cells_independent():
    """Distinct cells keep independent counts (no cross-cell contamination)."""
    from pokerl.agent.ppo import count_novelty_bonus
    d = {}
    a = (0x0C, 1, 1, (0, 0, 0, 0))
    b = (0x0C, 2, 2, (0, 0, 0, 0))
    count_novelty_bonus(d, a, 1.0)
    count_novelty_bonus(d, a, 1.0)
    assert count_novelty_bonus(d, b, 1.0) == 1.0   # b still fresh


def test_ppoconfig_accepts_count_mode():
    """Config plumbing: rnd_input='count' + count_coef are valid PPOConfig fields."""
    from pokerl.agent.ppo import PPOConfig
    cfg = PPOConfig(rnd_enabled=True, rnd_input="count", count_coef=1.0)
    assert cfg.rnd_input == "count"
    assert cfg.count_coef == 1.0
