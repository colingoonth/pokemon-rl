"""Smoke tests for the actor-critic network shapes."""
from __future__ import annotations

import torch

from pokerl.agent.networks import ActorCritic
from pokerl.env.pokemon_red_env import ACTIONS, SCREEN_H, SCREEN_W
from pokerl.env.ram_map import PROGRESS_DIM


def _prog(batch: int) -> torch.Tensor:
    """Zero story-progress bits (curriculum start) for `batch` rows."""
    return torch.zeros(batch, PROGRESS_DIM, dtype=torch.float32)


def test_forward_shapes_single_obs():
    frame_stack = 4
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    obs = torch.zeros(1, frame_stack, SCREEN_H, SCREEN_W, dtype=torch.uint8)
    logits, value, value_int = net(obs, _prog(1))
    assert logits.shape == (1, len(ACTIONS))
    assert value.shape == (1,)
    assert value_int.shape == (1,)


def test_forward_shapes_batched():
    frame_stack = 4
    batch = 8
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    obs = torch.randint(0, 256, (batch, frame_stack, SCREEN_H, SCREEN_W), dtype=torch.uint8)
    logits, value, value_int = net(obs, _prog(batch))
    assert logits.shape == (batch, len(ACTIONS))
    assert value.shape == (batch,)
    assert value_int.shape == (batch,)


def test_value_methods_match_forward():
    frame_stack = 4
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    net.eval()
    obs = torch.randint(0, 256, (3, frame_stack, SCREEN_H, SCREEN_W), dtype=torch.uint8)
    prog = _prog(3)
    with torch.no_grad():
        _, v_full, v_int_full = net(obs, prog)
        v_only = net.value(obs, prog)
        v_int_only = net.value_int(obs, prog)
    assert torch.allclose(v_full, v_only)
    assert torch.allclose(v_int_full, v_int_only)


def test_critic_int_is_separate_head():
    """The intrinsic head is a distinct parameter set from the extrinsic one."""
    net = ActorCritic((4, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    assert net.critic.weight.data_ptr() != net.critic_int.weight.data_ptr()


def test_actor_init_is_near_uniform():
    """Small actor init -> initial logits near 0 -> ~uniform action distribution."""
    frame_stack = 4
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    obs = torch.zeros(64, frame_stack, SCREEN_H, SCREEN_W, dtype=torch.uint8)
    with torch.no_grad():
        logits, _, _ = net(obs, _prog(64))
    # logits should be small in magnitude at init
    assert logits.abs().max() < 1.0, f"Initial logits too large: max={logits.abs().max():.3f}"


def test_progress_bits_reach_heads():
    """Step 0 wiring guard: changing the story-progress bits must change the
    policy/value output, else the fused bits aren't actually consumed."""
    net = ActorCritic((4, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    net.eval()
    obs = torch.zeros(1, 4, SCREEN_H, SCREEN_W, dtype=torch.uint8)
    with torch.no_grad():
        l0, v0, vi0 = net(obs, torch.zeros(1, PROGRESS_DIM))
        l1, v1, vi1 = net(obs, torch.ones(1, PROGRESS_DIM))
    assert not (torch.allclose(l0, l1) and torch.allclose(v0, v1)
                and torch.allclose(vi0, vi1)), "progress bits do not affect output"
