"""PPO trainer for PokemonRedEnv (vectorized envs).

Derived from the CleanRL ppo_atari.py implementation pattern (typed out
from scratch per clean-room rule, not copied). The env must be a
gymnasium SyncVectorEnv (or compatible) — n_envs=1 is a valid trivial
case but the loop is always shaped for parallelism.

References for anyone reading this later:
  - The 9 PPO implementation details that matter:
    Schulman et al. 2017 + the ICLR 2020 "Implementation Matters" paper.
  - GAE: Schulman et al. 2016.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import VectorEnv
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP

from pokerl.agent.networks import ActorCritic
from pokerl.infra import dist as dist_helpers
from pokerl.infra.logging import CSVLogger


@dataclass
class PPOConfig:
    total_timesteps: int = 10_000
    n_envs: int = 1               # vectorized env count
    n_steps: int = 128            # rollout length per env
    n_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1
    log_csv: str | None = None
    async_envs: bool = False      # multi-process envs to escape the GIL
    save_every: int = 100         # iterations between intermediate checkpoints; 0 disables
    reward_class: str = "RewardV0_2"  # which Reward subclass make_vec_env should instantiate
    state_path: str | None = None  # optional env start-state override; None = env default


@dataclass
class RolloutBuffer:
    """Per-iteration storage. Shapes are (n_steps, n_envs, ...)."""

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)

    @classmethod
    def empty(
        cls,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        device: str,
    ) -> "RolloutBuffer":
        return cls(
            obs=torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.uint8, device=device),
            actions=torch.zeros((n_steps, n_envs), dtype=torch.long, device=device),
            log_probs=torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device),
            values=torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device),
            rewards=torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device),
            dones=torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device),
        )

    def compute_gae(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute per-env GAE advantages + returns."""
        n_steps = self.rewards.shape[0]
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros_like(last_values)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_nonterminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        self.advantages = advantages
        self.returns = advantages + self.values


def train(
    env_fn: Callable[[], VectorEnv],
    cfg: PPOConfig,
    resume_path: Path | None = None,
) -> ActorCritic:
    # Distributed context — env-var-driven. world_size=1 = single-process path,
    # exactly the original behavior. >1 = torchrun-launched DDP.
    dctx = dist_helpers.get_dist_info()

    # Per-rank seeding so each rank's stochastic policy + env resets are
    # disjoint but deterministic.
    torch.manual_seed(cfg.seed + dctx.rank)
    np.random.seed(cfg.seed + dctx.rank)

    envs = env_fn()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    n_envs_local = envs.num_envs
    # cfg.n_envs is the GLOBAL convention (total across all ranks). The
    # train script is responsible for constructing env_fn with the per-rank
    # slice. Sanity-check here.
    expected_per_rank = cfg.n_envs // dctx.world_size
    assert n_envs_local == expected_per_rank, (
        f"env_fn produced {n_envs_local} envs/rank but expected "
        f"{expected_per_rank} (cfg.n_envs={cfg.n_envs}, world={dctx.world_size})"
    )
    assert cfg.n_envs % dctx.world_size == 0, (
        f"cfg.n_envs={cfg.n_envs} not divisible by world_size {dctx.world_size}"
    )
    n_envs_global = cfg.n_envs

    obs_shape: tuple[int, ...] = tuple(envs.single_observation_space.shape)
    n_actions = int(envs.single_action_space.n)

    if dctx.is_distributed:
        device = torch.device(f"cuda:{dctx.local_rank}")
    else:
        device = torch.device(cfg.device)

    raw_net = ActorCritic(obs_shape, n_actions=n_actions).to(device)
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        raw_net.load_state_dict(state)
        if dctx.is_main:
            print(f"Resumed weights from {resume_path}")

    # Wrap with DDP AFTER resume-load (else state-dict keys are 'module.<X>').
    if dctx.is_distributed:
        net: nn.Module = DDP(
            raw_net, device_ids=[dctx.local_rank], output_device=dctx.local_rank
        )
        # NCCL warmup: amortize the first-collective handshake outside the
        # per-iter timer so iter-1 isn't a 200-500ms outlier.
        import torch.distributed as _d
        _d.all_reduce(torch.zeros(1, device=device))
    else:
        net = raw_net

    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Disjoint per-rank env seeding. Spaced by 10k so reset seeds don't
    # collide across ranks for any realistic n_envs_local.
    obs_np, _ = envs.reset(seed=cfg.seed + dctx.rank * 10_000)
    obs_t = torch.from_numpy(obs_np).to(device)
    dones_t = torch.zeros(n_envs_local, dtype=torch.float32, device=device)

    # Per-rank batch sizes. minibatch_size in yaml is GLOBAL convention too.
    batch_size_local = cfg.n_steps * n_envs_local
    batch_size_global = batch_size_local * dctx.world_size
    assert cfg.minibatch_size % dctx.world_size == 0, (
        f"cfg.minibatch_size={cfg.minibatch_size} not divisible by "
        f"world_size {dctx.world_size}"
    )
    minibatch_size_local = cfg.minibatch_size // dctx.world_size
    assert batch_size_local % minibatch_size_local == 0, (
        f"batch_size_local {batch_size_local} not divisible by "
        f"minibatch_size_local {minibatch_size_local}"
    )

    n_iterations = max(1, cfg.total_timesteps // batch_size_global)
    global_step = 0  # counts GLOBAL samples (sum across all ranks)

    ep_returns_running = np.zeros(n_envs_local, dtype=np.float64)
    ep_lengths_running = np.zeros(n_envs_local, dtype=np.int64)
    finished_returns: list[float] = []

    # CSV + checkpoints are rank-0 only. Other ranks keep these None.
    csv_logger = CSVLogger(cfg.log_csv) if (dctx.is_main and cfg.log_csv) else None
    ckpt_dir: Path | None = None
    if dctx.is_main and cfg.save_every > 0 and cfg.log_csv:
        ckpt_dir = Path(cfg.log_csv).parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, n_iterations + 1):
        iter_t0 = time.perf_counter()
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1) / n_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.learning_rate

        buf = RolloutBuffer.empty(cfg.n_steps, n_envs_local, obs_shape, str(device))

        for step in range(cfg.n_steps):
            global_step += n_envs_global
            buf.obs[step] = obs_t
            buf.dones[step] = dones_t

            with torch.no_grad():
                logits, values = net(obs_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            buf.values[step] = values
            buf.actions[step] = actions
            buf.log_probs[step] = log_probs

            actions_np = actions.cpu().numpy()
            obs_np, rewards_np, term_np, trunc_np, _ = envs.step(actions_np)
            done_np = np.logical_or(term_np, trunc_np)

            buf.rewards[step] = torch.from_numpy(rewards_np).to(device).float()

            ep_returns_running += rewards_np
            ep_lengths_running += 1
            for i, d in enumerate(done_np):
                if d:
                    finished_returns.append(float(ep_returns_running[i]))
                    ep_returns_running[i] = 0.0
                    ep_lengths_running[i] = 0

            obs_t = torch.from_numpy(obs_np).to(device)
            dones_t = torch.from_numpy(done_np).to(device).float()

        with torch.no_grad():
            _, last_values_t = net(obs_t)

        buf.compute_gae(
            last_values=last_values_t,
            last_dones=dones_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        # Flatten (n_steps, n_envs_local, ...) -> (batch_size_local, ...)
        b_obs = buf.obs.reshape((batch_size_local, *obs_shape))
        b_actions = buf.actions.reshape(batch_size_local)
        b_log_probs = buf.log_probs.reshape(batch_size_local)
        b_advantages = buf.advantages.reshape(batch_size_local)
        b_returns = buf.returns.reshape(batch_size_local)
        b_values = buf.values.reshape(batch_size_local)

        indices = np.arange(batch_size_local)
        last_pg_loss = last_v_loss = last_entropy = 0.0
        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size_local, minibatch_size_local):
                mb_idx = indices[start : start + minibatch_size_local]
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_values_old = b_values[mb_idx]

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logits, values_new = net(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

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

        recent = finished_returns[-20:] if finished_returns else [float(ep_returns_running.mean())]
        local_mean_ret = float(np.mean(recent))
        local_episodes = len(finished_returns)

        # Local aggregate exploration stat from this rank's envs.
        try:
            tiles_per_env = envs.call("unique_tiles_visited")
            local_tiles = int(sum(tiles_per_env))
        except Exception:
            local_tiles = 0

        # Cross-rank aggregation: mean of per-rank means for returns,
        # sum-across-ranks for counts. Approximates global stats well when
        # ranks are balanced (same envs/rank, same step budget).
        mean_ret = dist_helpers.all_reduce_mean(local_mean_ret, device)
        episode_count = dist_helpers.all_reduce_sum_int(local_episodes, device)
        unique_tiles_total = dist_helpers.all_reduce_sum_int(local_tiles, device)

        iter_dt = time.perf_counter() - iter_t0
        sps = batch_size_global / iter_dt if iter_dt > 0 else 0.0

        if dctx.is_main and iteration % cfg.log_every == 0:
            ts = time.strftime("%H:%M:%S")
            print(
                f"[{ts}] iter {iteration:4d}  step {global_step:7d}  "
                f"mean_return(last20) {mean_ret:+.3f}  "
                f"episodes {episode_count:4d}  "
                f"tiles(all_envs) {unique_tiles_total:5d}  "
                f"pg_loss {last_pg_loss:+.4f}  "
                f"v_loss {last_v_loss:.4f}  "
                f"H {last_entropy:.3f}  "
                f"dt {iter_dt:.2f}s  "
                f"sps {sps:.0f}"
            )
        if csv_logger is not None:
            csv_logger.log({
                "iteration": iteration,
                "global_step": global_step,
                "mean_return_last20": mean_ret,
                "episodes_completed": episode_count,
                "unique_tiles_total": unique_tiles_total,
                "policy_loss": last_pg_loss,
                "value_loss": last_v_loss,
                "entropy": last_entropy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "iter_seconds": iter_dt,
                "samples_per_second": sps,
            })

        if ckpt_dir is not None and iteration % cfg.save_every == 0:
            path = ckpt_dir / f"iter_{iteration:06d}.pt"
            # Unwrap DDP for checkpoint compatibility with single-process load.
            state_dict = (
                net.module.state_dict() if dctx.is_distributed else net.state_dict()
            )
            torch.save(state_dict, path)
            print(f"  -> saved checkpoint {path}")

    if csv_logger is not None:
        csv_logger.close()
    envs.close()
    # Return the raw module so callers can torch.save without worrying about
    # the DDP wrapper.
    return net.module if dctx.is_distributed else net
