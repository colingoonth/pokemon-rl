"""Minimal single-env PPO trainer for PokemonRedEnv.

Derived from the implementation pattern in CleanRL ppo_atari.py
(typed out from scratch per clean-room rule, not copied). Single env,
no vectorization — vectorized envs land when we're ready to push to
ELSA. Goal here is a small, readable loop that can run for a few
hundred steps locally and demonstrably update the policy.

References for anyone reading this later:
  - The 9 PPO implementation details that matter:
    Schulman et al. 2017 + the ICLR 2020 "Implementation Matters" paper.
  - GAE: Schulman et al. 2016.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from pokerl.agent.networks import ActorCritic


@dataclass
class PPOConfig:
    total_timesteps: int = 10_000
    n_steps: int = 128            # rollout length
    n_epochs: int = 4             # update epochs per rollout
    minibatch_size: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.999          # high — Pokemon is long-horizon
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1            # iterations between log prints


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)

    @classmethod
    def empty(cls, n_steps: int, obs_shape: tuple[int, ...], device: str) -> "RolloutBuffer":
        return cls(
            obs=torch.zeros((n_steps, *obs_shape), dtype=torch.uint8, device=device),
            actions=torch.zeros(n_steps, dtype=torch.long, device=device),
            log_probs=torch.zeros(n_steps, dtype=torch.float32, device=device),
            values=torch.zeros(n_steps, dtype=torch.float32, device=device),
            rewards=torch.zeros(n_steps, dtype=torch.float32, device=device),
            dones=torch.zeros(n_steps, dtype=torch.float32, device=device),
        )

    def compute_gae(
        self, last_value: float, last_done: float, gamma: float, gae_lambda: float
    ) -> None:
        """Compute advantages with Generalized Advantage Estimation."""
        n = self.rewards.shape[0]
        advantages = torch.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_nonterminal = 1.0 - last_done
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1].item()
                next_value = self.values[t + 1].item()
            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_nonterminal
                - self.values[t].item()
            )
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        self.advantages = advantages
        self.returns = advantages + self.values


def train(env_fn: Callable[[], gym.Env], cfg: PPOConfig) -> ActorCritic:
    """Run PPO on a single env. Returns the trained network."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = env_fn()
    obs_shape = env.observation_space.shape
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    device = torch.device(cfg.device)
    net = ActorCritic(obs_shape, n_actions=n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs_np, _ = env.reset(seed=cfg.seed)
    obs_t = torch.from_numpy(obs_np).to(device)
    done = False
    global_step = 0
    n_iterations = max(1, cfg.total_timesteps // cfg.n_steps)

    ep_return = 0.0
    ep_len = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for iteration in range(1, n_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1) / n_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.learning_rate

        buf = RolloutBuffer.empty(cfg.n_steps, obs_shape, cfg.device)

        for step in range(cfg.n_steps):
            global_step += 1
            buf.obs[step] = obs_t
            buf.dones[step] = float(done)

            with torch.no_grad():
                logits, value = net(obs_t.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            buf.values[step] = value.squeeze(0)
            buf.actions[step] = action.squeeze(0)
            buf.log_probs[step] = log_prob.squeeze(0)

            obs_np, reward, term, trunc, _ = env.step(int(action.item()))
            done = bool(term or trunc)
            buf.rewards[step] = float(reward)

            ep_return += float(reward)
            ep_len += 1

            if done:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)
                ep_return = 0.0
                ep_len = 0
                obs_np, _ = env.reset()

            obs_t = torch.from_numpy(obs_np).to(device)

        # Bootstrap value of the final state
        with torch.no_grad():
            _, last_value_t = net(obs_t.unsqueeze(0))
        buf.compute_gae(
            last_value=float(last_value_t.item()),
            last_done=float(done),
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        # PPO update epochs
        flat_obs = buf.obs
        flat_actions = buf.actions
        flat_log_probs = buf.log_probs
        flat_advantages = buf.advantages
        flat_returns = buf.returns
        flat_values = buf.values

        indices = np.arange(cfg.n_steps)
        last_pg_loss = last_v_loss = last_entropy = 0.0
        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, cfg.n_steps, cfg.minibatch_size):
                mb_idx = indices[start : start + cfg.minibatch_size]
                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_adv = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_values_old = flat_values[mb_idx]

                # Normalize advantages within the minibatch (PPO best practice)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logits, values_new = net(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss
                v_clipped = mb_values_old + (values_new - mb_values_old).clamp(
                    -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_unclipped = (values_new - mb_returns).pow(2)
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_pg_loss = float(policy_loss.item())
                last_v_loss = float(value_loss.item())
                last_entropy = float(entropy.item())

        if iteration % cfg.log_every == 0:
            recent = episode_returns[-10:] if episode_returns else [ep_return]
            mean_ret = float(np.mean(recent))
            print(
                f"iter {iteration:4d}  step {global_step:6d}  "
                f"mean_return(last10) {mean_ret:+.3f}  "
                f"policy_loss {last_pg_loss:+.4f}  "
                f"value_loss {last_v_loss:.4f}  "
                f"entropy {last_entropy:.3f}"
            )

    env.close()
    return net
