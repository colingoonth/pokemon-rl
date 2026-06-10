"""Smoke tests for the actor-critic network shapes."""
from __future__ import annotations

import torch

from pokerl.agent.networks import ActorCritic
from pokerl.env.pokemon_red_env import ACTIONS, SCREEN_H, SCREEN_W


def test_forward_shapes_single_obs():
    frame_stack = 4
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    obs = torch.zeros(1, frame_stack, SCREEN_H, SCREEN_W, dtype=torch.uint8)
    logits, value, value_int = net(obs)
    assert logits.shape == (1, len(ACTIONS))
    assert value.shape == (1,)
    assert value_int.shape == (1,)


def test_forward_shapes_batched():
    frame_stack = 4
    batch = 8
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    obs = torch.randint(0, 256, (batch, frame_stack, SCREEN_H, SCREEN_W), dtype=torch.uint8)
    logits, value, value_int = net(obs)
    assert logits.shape == (batch, len(ACTIONS))
    assert value.shape == (batch,)
    assert value_int.shape == (batch,)


def test_value_methods_match_forward():
    frame_stack = 4
    net = ActorCritic((frame_stack, SCREEN_H, SCREEN_W), n_actions=len(ACTIONS))
    net.eval()
    obs = torch.randint(0, 256, (3, frame_stack, SCREEN_H, SCREEN_W), dtype=torch.uint8)
    with torch.no_grad():
        _, v_full, v_int_full = net(obs)
        v_only = net.value(obs)
        v_int_only = net.value_int(obs)
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
        logits, _, _ = net(obs)
    # logits should be small in magnitude at init
    assert logits.abs().max() < 1.0, f"Initial logits too large: max={logits.abs().max():.3f}"
