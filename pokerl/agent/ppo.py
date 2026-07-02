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

import math
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
from pokerl.agent.rnd import RewardForwardFilter, RNDModel, RunningMeanStd
from pokerl.infra import dist as dist_helpers
from pokerl.infra.logging import CSVLogger


def apply_truncation_bootstrap(
    rewards_step: torch.Tensor,
    term_np,
    trunc_np,
    info: dict,
    value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    gamma: float,
    n_envs: int,
    device: torch.device | str,
    progress_dim: int = 4,
) -> torch.Tensor:
    """Fold the time-limit bootstrap into truncated steps' rewards.

    An episode that ends on the step budget is TRUNCATED, not terminated: the
    future still has value, so PPO must bootstrap it rather than cut value to
    zero (the classic "truncation treated as termination" bug). Under
    SAME_STEP autoreset the returned obs is already the next episode's reset
    obs, so the genuine terminal observation arrives in info['final_obs'] with
    a boolean mask in info['_final_obs']. For each env that truncated (and did
    NOT truly terminate), we add `gamma * V(final_obs)` to that step's reward.
    compute_gae then masks the boundary as usual (next_nonterminal=0) and the
    folded term supplies the value that masking would otherwise discard.

    True terminations (term=True) get no bootstrap — a real terminal state has
    zero future value. This env never terminates, but the mask keeps it general.

    `value_fn` maps an obs batch to its (extrinsic) state-value estimate; passing
    it as a callable keeps this RND-ready (later: the extrinsic value head).
    Returns a new tensor; does not mutate the input.
    """
    fin_mask = np.asarray(info.get("_final_obs", np.zeros(n_envs, dtype=bool)))
    trunc_only = np.asarray(trunc_np) & ~np.asarray(term_np) & fin_mask
    if not trunc_only.any():
        return rewards_step
    idxs = np.nonzero(trunc_only)[0]
    final_obs_arr = np.stack([info["final_obs"][i] for i in idxs])
    # The value head now also consumes the story-progress bits, so bootstrap
    # V(final_obs) with the TERMINAL progress (from final_info), not the reset
    # episode's. Missing final_info progress falls back to zeros (curriculum
    # start has all bits clear anyway).
    fin_prog = info.get("final_info", {}).get("progress")
    if fin_prog is not None:
        final_prog_arr = np.stack([np.asarray(fin_prog[i], dtype=np.float32) for i in idxs])
    else:
        final_prog_arr = np.zeros((len(idxs), progress_dim), dtype=np.float32)
    with torch.no_grad():
        final_vals = value_fn(
            torch.from_numpy(final_obs_arr).to(device),
            torch.from_numpy(final_prog_arr).to(device),
        ).reshape(-1)
    idx_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
    out = rewards_step.clone()
    out[idx_t] = out[idx_t] + gamma * final_vals
    return out


def pad_progress_head_weights(net_state: dict, model_sd: dict) -> dict:
    """Warm-start surgery for V0.5.9 "Step 0" (fuse story-progress bits at the
    head input).

    The actor/critic head weights grew from (out, hidden) to (out, hidden +
    PROGRESS_DIM) when the progress bits were concatenated at the head input. A
    pre-Step-0 checkpoint carries the OLD narrow head weights; load_state_dict
    would reject them on a size mismatch (strict=False only tolerates
    missing/unexpected KEYS, not shape mismatches on shared keys).

    We RIGHT-PAD the new progress columns with ZERO. Zero columns mean the
    warm-started policy is BEHAVIORALLY IDENTICAL to the source at step 0
    (progress contributes nothing), and learns to use the bits from there —
    gradients flow into the zero columns the moment a story bit turns on. The
    visual trunk (backbone/fc) and all biases are unchanged, so they transfer
    verbatim. A checkpoint already at the new width (a V0.5.9 full-resume) has
    matching shapes and is left untouched. Mutates and returns net_state.
    """
    for k in ("actor.weight", "critic.weight", "critic_int.weight"):
        if k in net_state and k in model_sd and net_state[k].shape != model_sd[k].shape:
            src = net_state[k]
            dst = torch.zeros_like(model_sd[k])
            old_in = src.shape[1]
            if old_in > dst.shape[1]:
                raise RuntimeError(
                    f"checkpoint head {k} is WIDER than the model "
                    f"({tuple(src.shape)} vs {tuple(dst.shape)}) — not a Step-0 grow"
                )
            dst[:, :old_in] = src.to(dst.device, dst.dtype)
            net_state[k] = dst
    return net_state


def count_novelty_bonus(count_dict: dict, cell, coef: float) -> float:
    """Episodic count-based novelty for one env-step (V0.5.6).

    `cell` is the hashable key — here (map_id, x, y, progress_bits). Returns
    `coef / sqrt(N+1)` for the current per-episode visit count N of that cell,
    then increments the count in place. Bounded in (0, coef] and strictly
    decreasing in N, so revisits diminish and a genuinely fresh cell — including
    one freshly re-keyed by a story-flag flip (the Oak-parcel backtrack) — pays
    the maximum. The count dict is cleared per episode by the caller, which is
    what keeps this from PERMANENTLY saturating like global RND.
    """
    n = count_dict.get(cell, 0)
    count_dict[cell] = n + 1
    return coef / math.sqrt(n + 1)


@dataclass
class PPOConfig:
    total_timesteps: int = 10_000
    n_envs: int = 1               # vectorized env count
    n_steps: int = 128            # rollout length per env (PPO buffer, NOT episode len)
    max_steps: int = 4096         # episode truncation length (exploration horizon)
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
    frame_skip: int = 1            # repeat the action+24-frame loop N times per agent step

    # --- CPU throughput levers (default = prior behavior; see Day 10 sweep) ---
    torch_threads: int = 1         # intra-op ATen/BLAS threads for the MAIN process only
                                   # (policy forward + PPO update). Env workers stay
                                   # single-threaded via OMP_NUM_THREADS=1 in the env, set
                                   # before they spawn. >1 only helps the GIL-bound coordinator.
    vec_context: str = "spawn"     # AsyncVectorEnv start method. "spawn" is required when a
                                   # CUDA context exists (DDP); "fork" is cheaper IPC + startup
                                   # and safe on the GPU-less Genoa CPU nodes (no CUDA to re-init).

    # --- RND / curiosity (all defaulted so the plain-PPO path is unchanged) ---
    rnd_enabled: bool = False       # master switch; off = exact pre-RND behavior
    ext_coef: float = 2.0           # weight on extrinsic advantage
    int_coef: float = 1.0           # weight on intrinsic advantage (start value)
    # Curiosity -> exploitation handoff: linearly anneal int_coef from int_coef
    # to int_coef_final over training (same `frac` schedule as anneal_lr). Lets
    # RND bootstrap exploration early, then hand off to the (now dense) extrinsic
    # story signal so the policy commits to playing the game. ext_coef is held
    # fixed; as int_coef -> floor, coef_norm -> ext_coef and the combined
    # advantage collapses cleanly toward pure extrinsic. Off by default (exact
    # prior behavior); int_coef_final ignored unless anneal_int_coef is true.
    anneal_int_coef: bool = False
    int_coef_final: float = 0.0     # floor int_coef anneals toward
    int_gamma: float = 0.99         # intrinsic discount (separate, non-episodic stream)
    rnd_update_proportion: float = 0.25  # fraction of minibatch used to train the predictor
    rnd_feature_dim: int = 256      # target/predictor output width
    rnd_obs_norm_steps: int = 1024  # random pre-rollout steps to seed obs normalization
    rnd_obs_clip: float = 5.0       # clip normalized obs to +/- this
    rnd_int_clip: float = 5.0       # clip normalized intrinsic reward to +/- this
    # --- RND input mode (Day 11: animation-invariant curiosity) ---
    rnd_input: str = "frame"        # "frame" = conv-over-pixels (original, noisy-TV prone on
                                    # overworld animation); "ram" = map_id/x/y + progress, so
                                    # novelty is animation-invariant and a story-flag flip
                                    # refreshes it across the map (drives the Oak backtrack);
                                    # "count" = episodic count-based novelty (V0.5.6): NO neural
                                    # predictor — per-env per-episode visit counts of the cell
                                    # (map_id,x,y,progress_bits), bonus count_coef/sqrt(N+1),
                                    # reset each episode so it never PERMANENTLY saturates like
                                    # global RND. The progress_bits in the key re-light already-
                                    # walked tiles when a story flag flips (the Oak backtrack).
    rnd_num_maps: int = 256         # map_id embedding-table size (ram mode)
    rnd_n_fourier: int = 128        # random Fourier features for (x,y) (ram mode)
    rnd_fourier_scale: float = 16.0 # Fourier frequency scale = spatial-granularity knob (ram mode):
                                    # higher = sharper per-tile novelty, lower = smoother coverage
    count_coef: float = 1.0         # count-novelty bonus scale c in c/sqrt(N+1) (count mode).
                                    # Absolute value is washed out by per-stream advantage
                                    # normalization (cross-stream balance is int_coef's job), so
                                    # c=1 keeps the bonus bounded in (0,1]; it is a value-head
                                    # conditioning constant, not a balance knob.


@dataclass
class RolloutBuffer:
    """Per-iteration storage. Shapes are (n_steps, n_envs, ...).

    The `*_int` / `rnd_*` fields are only populated when RND is enabled; they
    default to None so the plain-PPO construction path is unchanged.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    # RND streams (None when rnd disabled)
    values_int: torch.Tensor | None = None
    rewards_int: torch.Tensor | None = None
    rnd_frame: torch.Tensor | None = None      # landed single frame (uint8) for predictor training (frame mode)
    rnd_state: torch.Tensor | None = None      # landed [map_id,x,y] (float32) for predictor training (ram mode)
    rnd_progress: torch.Tensor | None = None   # landed progress bits
    # Story-progress bits for the POLICY/VALUE net (V0.5.9 "Step 0"). Always
    # allocated (unlike rnd_progress, which is RND-path only): the net now
    # consumes these bits every step so the update must replay them.
    progress: torch.Tensor = field(init=False)
    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)
    advantages_int: torch.Tensor = field(init=False)
    returns_int: torch.Tensor = field(init=False)

    @classmethod
    def empty(
        cls,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        device: str,
        rnd: bool = False,
        progress_dim: int = 0,
        rnd_input: str = "frame",
        state_dim: int = 3,
        policy_progress_dim: int = 4,
    ) -> "RolloutBuffer":
        z2 = lambda: torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        buf = cls(
            obs=torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.uint8, device=device),
            actions=torch.zeros((n_steps, n_envs), dtype=torch.long, device=device),
            log_probs=z2(),
            values=z2(),
            rewards=z2(),
            dones=z2(),
        )
        # Policy-progress bits: one row per step, always present.
        buf.progress = torch.zeros(
            (n_steps, n_envs, policy_progress_dim), dtype=torch.float32, device=device
        )
        if rnd:
            buf.values_int = z2()
            buf.rewards_int = z2()
            # Count mode needs only the intrinsic value/reward streams — there is
            # no neural predictor to train, so no landed frame/state/progress
            # buffers are allocated.
            if rnd_input != "count":
                buf.rnd_progress = torch.zeros((n_steps, n_envs, progress_dim), dtype=torch.float32, device=device)
                if rnd_input == "ram":
                    buf.rnd_state = torch.zeros((n_steps, n_envs, state_dim), dtype=torch.float32, device=device)
                else:
                    h, w = obs_shape[-2], obs_shape[-1]
                    buf.rnd_frame = torch.zeros((n_steps, n_envs, 1, h, w), dtype=torch.uint8, device=device)
        return buf

    @staticmethod
    def _gae(rewards, values, dones, last_values, last_dones, gamma, lam, episodic):
        """Generalized Advantage Estimation. episodic=True masks at episode
        boundaries (extrinsic stream); episodic=False never masks — the
        intrinsic novelty stream is non-episodic, it flows across resets."""
        n_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_values)
        for t in reversed(range(n_steps)):
            if episodic:
                nonterminal = (1.0 - last_dones) if t == n_steps - 1 else (1.0 - dones[t + 1])
            else:
                nonterminal = torch.ones_like(last_values)
            next_values = last_values if t == n_steps - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_values * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae
        return advantages, advantages + values

    def compute_gae(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Extrinsic (episodic) GAE advantages + returns."""
        self.advantages, self.returns = self._gae(
            self.rewards, self.values, self.dones,
            last_values, last_dones, gamma, gae_lambda, episodic=True,
        )

    def compute_gae_intrinsic(
        self,
        last_values_int: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Intrinsic (NON-episodic) GAE advantages + returns. The novelty
        stream ignores episode boundaries — curiosity is a property of the
        agent's lifelong experience, not of any single episode."""
        zero_dones = torch.zeros_like(last_values_int)
        self.advantages_int, self.returns_int = self._gae(
            self.rewards_int, self.values_int, self.dones,
            last_values_int, zero_dones, gamma, gae_lambda, episodic=False,
        )


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
    # Bump MAIN-process intra-op threads AFTER env_fn() — by now all async
    # workers have spawned/forked inheriting OMP_NUM_THREADS=1, so they stay
    # single-threaded while the GIL-bound coordinator (policy forward + PPO
    # update) gets more cores. Setting this earlier would fan the workers too.
    if cfg.torch_threads != 1:
        torch.set_num_threads(cfg.torch_threads)
        if dctx.is_main:
            print(f"main-process torch threads: {torch.get_num_threads()}")
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

    # Policy/value net now fuses the story-progress bits at the head input
    # (V0.5.9 "Step 0"): the pixels can't show whether the parcel is held, so
    # the net was blind to the fact that flips the optimal direction at Oak.
    from pokerl.env import ram_map as _ram_map
    policy_progress_dim = _ram_map.PROGRESS_DIM
    raw_net = ActorCritic(
        obs_shape, n_actions=n_actions, progress_dim=policy_progress_dim
    ).to(device)
    resume_ckpt: dict | None = None  # set iff a FULL-resume dict was loaded
    resume_global_step = 0
    resume_iteration = 0
    if resume_path is not None:
        # weights_only=False: a full-resume checkpoint carries optimizer state,
        # numpy normalization stats, and python scalars, not just tensors. These
        # are our own trusted checkpoints.
        loaded = torch.load(resume_path, map_location=device, weights_only=False)
        # Two resume modes, distinguished by the checkpoint shape:
        #   FULL resume      — a V0.5 dict {"net", "optimizer", "rnd", ...}:
        #                      continue the run (weights + optimizer + curiosity
        #                      + step/anneal position). For chaining reservations.
        #   WEIGHTS-ONLY     — a bare state_dict (e.g. a pre-RND 2-head
        #   warm-start         checkpoint): transfer policy weights only;
        #                      curiosity / optimizer / step counter start fresh.
        #                      This is the clean warm-vs-cold A/B contract.
        if isinstance(loaded, dict) and "net" in loaded:
            resume_ckpt = loaded
            net_state = loaded["net"]
        else:
            net_state = loaded
        # V0.5.9 "Step 0": zero-pad a pre-Step-0 checkpoint's narrow head weights
        # up to the progress-fused width, so the warm-start starts identical to
        # the source policy and learns to use the story bits (see the helper).
        net_state = pad_progress_head_weights(net_state, raw_net.state_dict())
        # strict=False so a warm-start from a checkpoint that predates a head
        # (e.g. a 2-head pre-RND checkpoint has no critic_int) loads cleanly —
        # the missing head inits fresh. An UNEXPECTED key, by contrast, means
        # the checkpoint doesn't match this architecture: a real error.
        missing, unexpected = raw_net.load_state_dict(net_state, strict=False)
        if unexpected:
            raise RuntimeError(f"resume checkpoint has unexpected keys: {unexpected}")
        if dctx.is_main:
            mode = "full-resume" if resume_ckpt is not None else "weights-only warm-start"
            extra = f"  (fresh-init: {sorted(missing)})" if missing else ""
            print(f"Resumed weights from {resume_path} [{mode}]{extra}")

    # RND is single-process only for now: the predictor would need its own DDP
    # wrap + grad all-reduce, which the 1-GPU VCL runs don't need. Guard loudly.
    if cfg.rnd_enabled and dctx.is_distributed:
        raise NotImplementedError(
            "RND (rnd_enabled=True) is single-process only; do not launch under DDP."
        )

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

    # --- RND (curiosity) setup ---
    rnd_model: RNDModel | None = None
    obs_rms = reward_rms = rff = None
    count_dicts: list[dict] | None = None   # count mode: per-env per-episode visit counts
    progress_dim = 0
    if cfg.rnd_enabled:
        from pokerl.env import ram_map as _rm
        if cfg.rnd_input not in ("frame", "ram", "count"):
            raise ValueError(
                f"unknown rnd_input {cfg.rnd_input!r} (expected 'frame', 'ram', or 'count')"
            )
        if cfg.rnd_input == "count":
            # Count-based episodic novelty (V0.5.6): no neural predictor, no
            # obs/return normalization. One visit-count dict per env, cleared on
            # that env's episode boundary; the bonus is computed directly in the
            # rollout loop. progress_dim stays 0 (no predictor buffers).
            count_dicts = [dict() for _ in range(n_envs_local)]
        else:
            progress_dim = _rm.PROGRESS_DIM
            frame_shape = (1, obs_shape[-2], obs_shape[-1])
            rnd_model = RNDModel(
                frame_shape, progress_dim, feature_dim=cfg.rnd_feature_dim,
                input_mode=cfg.rnd_input, num_maps=cfg.rnd_num_maps,
                n_fourier=cfg.rnd_n_fourier, fourier_scale=cfg.rnd_fourier_scale,
            ).to(device)
            # obs_rms only normalizes the frame path. In "ram" mode the encoder does
            # its own deterministic featurization (embedding + Fourier on bounded
            # RAM values), so the running per-pixel normalizer is unused.
            if cfg.rnd_input == "frame":
                obs_rms = RunningMeanStd(shape=frame_shape)   # per-pixel obs normalization
            reward_rms = RunningMeanStd(shape=())             # intrinsic-return std
            rff = RewardForwardFilter(cfg.int_gamma, n_envs_local)

    # The predictor trains alongside the policy (target is frozen). One Adam
    # over both keeps the loop simple; LR annealing applies to all groups.
    params = list(net.parameters())
    if rnd_model is not None:
        params += list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(params, lr=cfg.learning_rate, eps=1e-5)

    # --- Full-resume restore (now that optimizer + RND objects exist) ---
    # Without restoring these a "resumed" run silently re-randomizes the RND
    # target (a brand-new novelty function), re-zeros obs/return normalization,
    # discards optimizer momentum, and re-anneals LR from peak — i.e. it does
    # NOT continue the run. Restore them so cross-reservation chaining is real.
    rnd_state_restored = False
    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        resume_global_step = int(resume_ckpt.get("global_step", 0))
        resume_iteration = int(resume_ckpt.get("iteration", 0))
        if cfg.rnd_enabled and "rnd" in resume_ckpt:
            rnd_model.load_state_dict(resume_ckpt["rnd"])
            if obs_rms is not None and resume_ckpt.get("obs_rms") is not None:
                o = resume_ckpt["obs_rms"]
                obs_rms.mean, obs_rms.var, obs_rms.count = o["mean"], o["var"], o["count"]
            r = resume_ckpt["reward_rms"]
            reward_rms.mean, reward_rms.var, reward_rms.count = r["mean"], r["var"], r["count"]
            saved_rff = np.asarray(resume_ckpt["rff"], dtype=np.float64)
            # rff is per-env; a length mismatch would silently broadcast (e.g.
            # saved n_envs=1 into n_envs=N) and corrupt the discounted-return
            # state. Refuse rather than continue with garbage normalization.
            if saved_rff.shape != rff.rewems.shape:
                raise RuntimeError(
                    f"resume rff shape {saved_rff.shape} != current "
                    f"{rff.rewems.shape}: n_envs changed between save and resume; "
                    f"chained runs must keep n_envs constant."
                )
            rff.rewems = saved_rff
            rnd_state_restored = True
        elif cfg.rnd_enabled and dctx.is_main:
            print("  NOTE: full-resume checkpoint carries no RND state; curiosity starts fresh.")
        elif (not cfg.rnd_enabled) and "rnd" in resume_ckpt and dctx.is_main:
            print("  WARNING: checkpoint carries RND state but rnd_enabled=False; "
                  "curiosity is discarded for this run.")
        saved_total = resume_ckpt.get("total_timesteps")
        if (saved_total is not None and saved_total != cfg.total_timesteps
                and dctx.is_main):
            print(f"  WARNING: resume total_timesteps {cfg.total_timesteps} != "
                  f"saved {saved_total}; LR-anneal slope will differ from the "
                  f"original run (intended only when deliberately extending).")
        if dctx.is_main:
            print(f"  full-resume: global_step={resume_global_step}, "
                  f"resuming after iteration {resume_iteration}")

    # Disjoint per-rank env seeding. Spaced by 10k so reset seeds don't
    # collide across ranks for any realistic n_envs_local.
    obs_np, reset_info = envs.reset(seed=cfg.seed + dctx.rank * 10_000)

    # RND obs-norm warmup: seed obs_rms from random rollouts so the first
    # intrinsic rewards aren't computed against uninitialized statistics, then
    # re-reset so training episodes start cleanly at the curriculum state.
    if (cfg.rnd_enabled and cfg.rnd_input == "frame"
            and cfg.rnd_obs_norm_steps > 0 and not rnd_state_restored):
        for _ in range(cfg.rnd_obs_norm_steps):
            a = np.array([envs.single_action_space.sample() for _ in range(n_envs_local)])
            o, _, _, _, _ = envs.step(a)
            obs_rms.update(o[:, -1:, :, :].astype(np.float64))
        obs_np, reset_info = envs.reset(seed=cfg.seed + dctx.rank * 10_000 + 1)
        if dctx.is_main:
            print(f"RND obs-norm warmup done ({cfg.rnd_obs_norm_steps} steps).")

    obs_t = torch.from_numpy(obs_np).to(device)
    dones_t = torch.zeros(n_envs_local, dtype=torch.float32, device=device)
    # Running policy-progress bits paired with obs_t (updated post-step below,
    # mirroring obs_t = obs_np). Seed from reset info; fall back to zeros (the
    # curriculum start state has every story bit clear).
    _rp = reset_info.get("progress") if isinstance(reset_info, dict) else None
    if _rp is not None:
        prog_t = torch.as_tensor(np.asarray(_rp, dtype=np.float32), device=device)
    else:
        prog_t = torch.zeros((n_envs_local, policy_progress_dim),
                             dtype=torch.float32, device=device)

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
    # global_step / start iteration continue from a full-resume (else 0/fresh).
    global_step = resume_global_step  # counts GLOBAL samples (sum across all ranks)
    # A resume whose total_timesteps yields n_iterations <= the saved iteration
    # would run the training loop ZERO times and then save a "final" checkpoint —
    # silently burning a reservation while looking successful. Fail loudly.
    if resume_iteration >= n_iterations:
        raise RuntimeError(
            f"resume_iteration={resume_iteration} >= n_iterations={n_iterations}: "
            f"this run would do no training. Increase total_timesteps "
            f"(current {cfg.total_timesteps}) to extend the run past the "
            f"checkpoint, or resume from an earlier checkpoint."
        )

    ep_returns_running = np.zeros(n_envs_local, dtype=np.float64)
    ep_lengths_running = np.zeros(n_envs_local, dtype=np.int64)
    finished_returns: list[float] = []

    # CSV + checkpoints are rank-0 only. Other ranks keep these None.
    csv_logger = CSVLogger(cfg.log_csv) if (dctx.is_main and cfg.log_csv) else None
    ckpt_dir: Path | None = None
    if dctx.is_main and cfg.save_every > 0 and cfg.log_csv:
        ckpt_dir = Path(cfg.log_csv).parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _build_ckpt(iteration: int, stats: dict | None = None) -> dict:
        """Full checkpoint so a resume CONTINUES the run (weights + optimizer +
        curiosity + step/anneal) rather than re-randomizing. watch.py/rollout.py
        and the weights-only warm-start path read the "net" sub-key; bare legacy
        state_dicts (no "net") still load too.

        ``stats`` is the snapshot of training metrics (entropy, return, lr, ...)
        AT THIS checkpoint, embedded so a single rsync'd .pt is self-describing —
        lets us pick which entropy level to warm-start from later without
        cross-referencing metrics.csv. Additive key; absent on legacy ckpts."""
        net_sd = net.module.state_dict() if dctx.is_distributed else net.state_dict()
        ckpt = {
            "net": net_sd,
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "iteration": iteration,
            "total_timesteps": cfg.total_timesteps,
            "stats": stats or {},
        }
        if cfg.rnd_enabled and rnd_model is not None:
            # The frozen target net IS the novelty function — persist it, the
            # predictor's progress, and both running normalizers, or a resume
            # silently restarts exploration from a different random target.
            # (Count mode has no neural RND / normalizers — its visit counts are
            # per-episode ephemeral, so there is nothing to persist.)
            ckpt["rnd"] = rnd_model.state_dict()
            ckpt["obs_rms"] = None if obs_rms is None else {
                "mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count,
            }
            ckpt["reward_rms"] = {
                "mean": reward_rms.mean, "var": reward_rms.var, "count": reward_rms.count,
            }
            ckpt["rff"] = rff.rewems
        return ckpt

    def _current_stats() -> dict:
        """Snapshot of the latest logged training metrics, to embed in a
        checkpoint. Closes over the loop locals; only call after the loop body
        has assigned them at least once. RND-only fields are guarded."""
        s = {
            "global_step": global_step,
            "mean_return_last20": mean_ret,
            "entropy": last_entropy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "episodes_completed": episode_count,
            "unique_tiles_total": unique_tiles_total,
        }
        if cfg.rnd_enabled:
            s["mean_int_reward"] = mean_int_reward
            s["int_coef_now"] = int_coef_now
        return s

    for iteration in range(resume_iteration + 1, n_iterations + 1):
        iter_t0 = time.perf_counter()
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1) / n_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.learning_rate

        # Curiosity -> exploitation handoff: linearly anneal int_coef from
        # cfg.int_coef down to cfg.int_coef_final over training (same frac
        # schedule as anneal_lr). RND bootstraps exploration early; as int_coef
        # -> floor, coef_norm -> ext_coef and the combined advantage collapses
        # toward pure extrinsic (the now-dense story signal). Off => constant.
        if cfg.anneal_int_coef:
            int_frac = 1.0 - (iteration - 1) / n_iterations
            int_coef_now = cfg.int_coef_final + int_frac * (cfg.int_coef - cfg.int_coef_final)
        else:
            int_coef_now = cfg.int_coef

        buf = RolloutBuffer.empty(
            cfg.n_steps, n_envs_local, obs_shape, str(device),
            rnd=cfg.rnd_enabled, progress_dim=progress_dim,
            rnd_input=cfg.rnd_input, policy_progress_dim=policy_progress_dim,
        )

        rollout_t0 = time.perf_counter()
        for step in range(cfg.n_steps):
            global_step += n_envs_global
            buf.obs[step] = obs_t
            buf.progress[step] = prog_t
            buf.dones[step] = dones_t

            with torch.no_grad():
                logits, values, values_int = net(obs_t, prog_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            buf.values[step] = values
            if cfg.rnd_enabled:
                buf.values_int[step] = values_int
            buf.actions[step] = actions
            buf.log_probs[step] = log_probs

            actions_np = actions.cpu().numpy()
            obs_np, rewards_np, term_np, trunc_np, info = envs.step(actions_np)
            done_np = np.logical_or(term_np, trunc_np)

            rewards_step = torch.from_numpy(rewards_np).to(device).float()
            # Truncation bootstrap (fixes truncation-as-termination): fold
            # gamma * V(final_obs) into truncated steps' rewards so compute_gae's
            # boundary mask doesn't throw away the cut future. See the helper.
            rewards_step = apply_truncation_bootstrap(
                rewards_step, term_np, trunc_np, info,
                value_fn=lambda o, p: net(o, p)[1],
                gamma=cfg.gamma, n_envs=n_envs_local, device=device,
                progress_dim=policy_progress_dim,
            )

            buf.rewards[step] = rewards_step

            ep_returns_running += rewards_np
            ep_lengths_running += 1
            for i, d in enumerate(done_np):
                if d:
                    finished_returns.append(float(ep_returns_running[i]))
                    ep_returns_running[i] = 0.0
                    ep_lengths_running[i] = 0

            obs_t = torch.from_numpy(obs_np).to(device)
            dones_t = torch.from_numpy(done_np).to(device).float()
            # Advance the running policy-progress to match the new obs_t. Under
            # SAME_STEP autoreset a done env's info["progress"] is already the
            # reset episode's bits, which correctly pairs with obs_np — same
            # convention as obs_t above. (The RND-landed novelty below uses the
            # final-substituted prog_np instead; these are deliberately distinct.)
            prog_t = torch.as_tensor(
                np.asarray(info["progress"], dtype=np.float32), device=device
            )

            # --- Intrinsic (RND) reward on the LANDED observation ---
            if cfg.rnd_enabled:
                # "Landed" = the state the action actually produced. Under
                # SAME_STEP autoreset a truncated env's returned obs/info are
                # already the RESET episode, so for those envs the genuine
                # terminal frame/progress/in_battle live in final_obs/final_info.
                # Substitute them (mirrors the extrinsic truncation bootstrap) so
                # novelty, the predictor target, and the battle mask all use the
                # real landed state — not the next episode's start.
                count_mode = cfg.rnd_input == "count"
                # Both "ram" and "count" key novelty on the [map_id,x,y] state
                # vector + progress bits; only "frame" uses the pixel path.
                state_mode = cfg.rnd_input in ("ram", "count")
                landed = obs_np[:, -1:, :, :].copy()             # (n_envs,1,H,W); frame mode
                prog_np = np.asarray(info["progress"], dtype=np.float32).copy()
                inbatt = np.asarray(info["in_battle"]).copy()
                if state_mode:
                    state_np = np.asarray(info["rnd_state"], dtype=np.float32).copy()
                fmask = np.asarray(info.get("_final_obs", np.zeros(n_envs_local, dtype=bool)))
                if fmask.any():
                    fin = info.get("final_info", {})
                    fin_prog = fin.get("progress")
                    fin_batt = fin.get("in_battle")
                    fin_state = fin.get("rnd_state")
                    for i in np.nonzero(fmask)[0]:
                        landed[i, 0] = info["final_obs"][i][-1]   # last frame of terminal stack
                        if fin_prog is not None:
                            prog_np[i] = np.asarray(fin_prog[i], dtype=np.float32)
                        if fin_batt is not None:
                            inbatt[i] = fin_batt[i]
                        if state_mode and fin_state is not None:
                            state_np[i] = np.asarray(fin_state[i], dtype=np.float32)
                if count_mode:
                    # Count-based episodic novelty. Bonus = count_coef/sqrt(N+1)
                    # on the per-env per-episode visit count of the cell key
                    # (map_id, x, y, progress_bits). progress_bits in the key is
                    # what re-lights already-walked tiles the moment a story flag
                    # flips (the Oak-parcel backtrack), and the per-episode reset
                    # (below) is what stops it PERMANENTLY saturating like global
                    # RND. Battle steps pay 0 (x,y frozen + unlearnable RNG).
                    nov = np.zeros(n_envs_local, dtype=np.float32)
                    for i in range(n_envs_local):
                        if inbatt[i] > 0:
                            continue
                        m, x, y = state_np[i]
                        key = (int(m), int(x), int(y),
                               tuple(int(v) for v in prog_np[i]))
                        nov[i] = count_novelty_bonus(count_dicts[i], key, cfg.count_coef)
                else:
                    # Build the RND observation: raw state vector (ram mode) or the
                    # running-normalized + clipped frame (frame mode).
                    if state_mode:
                        rnd_obs = torch.as_tensor(state_np, dtype=torch.float32, device=device)
                    else:
                        obs_rms.update(landed.astype(np.float64))
                        fn = np.clip(
                            (landed - obs_rms.mean) / (obs_rms.std + 1e-8),
                            -cfg.rnd_obs_clip, cfg.rnd_obs_clip,
                        )
                        rnd_obs = torch.as_tensor(fn, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        nov = rnd_model.novelty(
                            rnd_obs,
                            torch.as_tensor(prog_np, device=device),
                        ).cpu().numpy()
                # Battle-mask: battle RNG is unlearnable noisy-TV; pay no
                # curiosity for landing in / sitting through a battle. (Count mode
                # already skipped battle steps above; this stays idempotent.)
                nov = np.asarray(nov, dtype=np.float32)
                nov[inbatt > 0] = 0.0
                buf.rewards_int[step] = torch.as_tensor(nov, dtype=torch.float32, device=device)
                if count_mode:
                    # Reset each finished env's count dict AFTER its terminal step
                    # is counted, so the next (reset) step starts fresh — this is
                    # the per-episode reset that revives novelty on the return leg.
                    for i in np.nonzero(done_np)[0]:
                        count_dicts[i].clear()
                elif state_mode:
                    buf.rnd_state[step] = torch.as_tensor(state_np, dtype=torch.float32, device=device)
                    buf.rnd_progress[step] = torch.as_tensor(prog_np, device=device)
                else:
                    buf.rnd_frame[step] = torch.as_tensor(landed, dtype=torch.uint8, device=device)
                    buf.rnd_progress[step] = torch.as_tensor(prog_np, device=device)

        # Rollout = env-collection half (forward + envs.step + RND novelty). The
        # remainder of the iter is the update half (GAE + n_epochs of SGD). The
        # split tells us which half the CPU bottleneck lives in: rollout-bound =>
        # attack env coordination (n_envs / fork / frame_skip); update-bound =>
        # attack torch_threads / minibatch.
        rollout_dt = time.perf_counter() - rollout_t0

        with torch.no_grad():
            _, last_values_t, last_values_int_t = net(obs_t, prog_t)

        buf.compute_gae(
            last_values=last_values_t,
            last_dones=dones_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        mean_int_reward = 0.0
        mean_raw_novelty = 0.0
        if cfg.rnd_enabled:
            ri = buf.rewards_int.cpu().numpy()                  # (n_steps, n_envs)
            # Raw (pre-normalization) novelty: THIS is the curiosity-decay signal.
            mean_raw_novelty = float(ri.mean())
            if cfg.rnd_input == "count":
                # BYPASS the shared intrinsic return-RMS: the count bonus is
                # already bounded in (0, count_coef], and per-stream advantage
                # normalization sets cross-stream balance via int_coef. Running
                # the shared RMS here would let heterogeneous per-episode coverage
                # (esp. under a state-curriculum) jerk the reward scale around for
                # no benefit. Feed the raw bonus straight to intrinsic GAE.
                mean_int_reward = float(buf.rewards_int.mean().item())
            else:
                # Normalize intrinsic rewards by the running std of the discounted
                # intrinsic RETURNS (keeps novelty scale stable as the predictor
                # learns), clip, then non-episodic GAE.
                disc_returns = np.array([rff.update(ri[t]) for t in range(ri.shape[0])])
                reward_rms.update(disc_returns.reshape(-1))
                ri_norm = np.clip(ri / (np.sqrt(reward_rms.var) + 1e-8),
                                  -cfg.rnd_int_clip, cfg.rnd_int_clip)
                buf.rewards_int = torch.as_tensor(ri_norm, dtype=torch.float32, device=device)
                mean_int_reward = float(buf.rewards_int.mean().item())
            buf.compute_gae_intrinsic(last_values_int_t, cfg.int_gamma, cfg.gae_lambda)

        # Flatten (n_steps, n_envs_local, ...) -> (batch_size_local, ...)
        b_obs = buf.obs.reshape((batch_size_local, *obs_shape))
        b_progress = buf.progress.reshape((batch_size_local, policy_progress_dim))
        b_actions = buf.actions.reshape(batch_size_local)
        b_log_probs = buf.log_probs.reshape(batch_size_local)
        b_advantages = buf.advantages.reshape(batch_size_local)
        b_returns = buf.returns.reshape(batch_size_local)
        b_values = buf.values.reshape(batch_size_local)
        if cfg.rnd_enabled:
            b_adv_int = buf.advantages_int.reshape(batch_size_local)
            b_ret_int = buf.returns_int.reshape(batch_size_local)
            # Predictor-training buffers exist only when there's a neural RND
            # (frame/ram). Count mode has none — only the intrinsic value head.
            if rnd_model is not None:
                b_rnd_prog = buf.rnd_progress.reshape((batch_size_local, progress_dim))
                if cfg.rnd_input == "ram":
                    b_rnd_state = buf.rnd_state.reshape((batch_size_local, buf.rnd_state.shape[-1]))
                else:
                    fh, fw = obs_shape[-2], obs_shape[-1]
                    b_rnd_frame = buf.rnd_frame.reshape((batch_size_local, 1, fh, fw))
                    obs_mean_t = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
                    obs_std_t = torch.as_tensor(obs_rms.std, dtype=torch.float32, device=device)

        # Raw per-stream advantage scales (pre per-minibatch normalization). The
        # gap shows what ext_coef:int_coef is really weighting against.
        adv_ext_std = float(b_advantages.std().item())
        adv_int_std = float(b_adv_int.std().item()) if cfg.rnd_enabled else 0.0

        indices = np.arange(batch_size_local)
        last_pg_loss = last_v_loss = last_entropy = 0.0
        last_rnd_loss = last_v_int_loss = 0.0
        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size_local, minibatch_size_local):
                mb_idx = indices[start : start + minibatch_size_local]
                mb_obs = b_obs[mb_idx]
                mb_progress = b_progress[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_values_old = b_values[mb_idx]

                # Combine extrinsic + intrinsic advantage (RND). Normalize EACH
                # stream to unit std FIRST, then weight: otherwise the two streams
                # sit on different scales and ext_coef:int_coef is not the actual
                # ratio of pull. Per-stream normalization makes the coefficients
                # mean what they say. Plain-PPO path keeps the single combined norm.
                if cfg.rnd_enabled:
                    a_ext = b_advantages[mb_idx]
                    a_int = b_adv_int[mb_idx]
                    a_ext = (a_ext - a_ext.mean()) / (a_ext.std() + 1e-8)
                    a_int = (a_int - a_int.mean()) / (a_int.std() + 1e-8)
                    # Divide by the coefficient-vector norm so the COMBINED
                    # advantage stays ~unit scale (matching plain-PPO and prior
                    # runs) while ext_coef:int_coef stays the true ratio of pull.
                    # Without this, two unit-std streams at 2:1 give std
                    # ~sqrt(5)=2.24 — a silent ~2.2x policy-step inflation.
                    coef_norm = (cfg.ext_coef ** 2 + int_coef_now ** 2) ** 0.5
                    mb_adv = (cfg.ext_coef * a_ext
                              + int_coef_now * a_int) / (coef_norm + 1e-8)
                else:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logits, values_new, values_int_new = net(mb_obs, mb_progress)
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

                # Intrinsic value loss (simple MSE) + RND predictor loss.
                rnd_loss = torch.zeros((), device=device)
                if cfg.rnd_enabled:
                    v_int_loss = 0.5 * (values_int_new - b_ret_int[mb_idx]).pow(2).mean()
                    value_loss = value_loss + v_int_loss
                    last_v_int_loss = float(v_int_loss.item())
                    # Train the predictor on a random subset (update_proportion)
                    # of the minibatch's landed observations (normalized as in
                    # rollout for frame mode; raw state vector for ram mode).
                    # Count mode has no predictor — only the intrinsic value head
                    # above is trained; rnd_loss stays 0.
                    if rnd_model is not None:
                        if cfg.rnd_input == "ram":
                            rnd_obs_mb = b_rnd_state[mb_idx]
                        else:
                            fr = b_rnd_frame[mb_idx].float()
                            rnd_obs_mb = ((fr - obs_mean_t) / (obs_std_t + 1e-8)).clamp(
                                -cfg.rnd_obs_clip, cfg.rnd_obs_clip
                            )
                        nov_train = rnd_model.novelty(rnd_obs_mb, b_rnd_prog[mb_idx])
                        keep = (torch.rand(nov_train.shape[0], device=device)
                                < cfg.rnd_update_proportion).float()
                        rnd_loss = (nov_train * keep).sum() / (keep.sum() + 1e-8)
                        last_rnd_loss = float(rnd_loss.item())

                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy + rnd_loss)

                optimizer.zero_grad()
                loss.backward()
                # Separate grad-norm budgets for policy vs RND predictor: a
                # shared clip lets a large early predictor gradient eat into the
                # policy's norm budget (throttling the policy update) and vice
                # versa. Clip each group to its own max_grad_norm.
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                if rnd_model is not None:
                    nn.utils.clip_grad_norm_(
                        rnd_model.predictor.parameters(), cfg.max_grad_norm
                    )
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
            rnd_str = (
                f"int_r {mean_int_reward:+.4f}  nov_raw {mean_raw_novelty:.4f}  "
                f"rnd_loss {last_rnd_loss:.4f}  advσ_e/i {adv_ext_std:.2f}/{adv_int_std:.2f}  "
                if cfg.rnd_enabled else ""
            )
            print(
                f"[{ts}] iter {iteration:4d}  step {global_step:7d}  "
                f"mean_return(last20) {mean_ret:+.3f}  "
                f"episodes {episode_count:4d}  "
                f"tiles(all_envs) {unique_tiles_total:5d}  "
                f"{rnd_str}"
                f"pg_loss {last_pg_loss:+.4f}  "
                f"v_loss {last_v_loss:.4f}  "
                f"H {last_entropy:.3f}  "
                f"roll {rollout_dt:.1f}s  upd {iter_dt - rollout_dt:.1f}s  "
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
                "mean_int_reward": mean_int_reward,
                "mean_raw_novelty": mean_raw_novelty,
                "rnd_loss": last_rnd_loss,
                "value_int_loss": last_v_int_loss,
                "adv_ext_std": adv_ext_std,
                "adv_int_std": adv_int_std,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "iter_seconds": iter_dt,
                "rollout_seconds": rollout_dt,
                "update_seconds": iter_dt - rollout_dt,
                "samples_per_second": sps,
            })

        # Checkpoint on the cadence AND always on the final iteration, so the
        # last save_every-remainder iterations aren't stranded in the bare
        # final.pt (which would silently demote a resume to weights-only).
        # Checkpoint on the cadence AND always on the final iteration, so the
        # last save_every-remainder iterations aren't stranded in final.pt.
        if ckpt_dir is not None and (
            iteration % cfg.save_every == 0 or iteration == n_iterations
        ):
            path = ckpt_dir / f"iter_{iteration:06d}.pt"
            torch.save(_build_ckpt(iteration, _current_stats()), path)
            print(f"  -> saved checkpoint {path}")

    # Final checkpoint as a FULL dict so resuming from the canonical final.pt
    # continues the run instead of silently downgrading to a weights-only
    # warm-start. (run_dir is the metrics.csv parent.)
    if dctx.is_main and cfg.log_csv:
        final_path = Path(cfg.log_csv).parent / "final.pt"
        torch.save(_build_ckpt(iteration, _current_stats()), final_path)
        print(f"Saved final checkpoint -> {final_path}")

    if csv_logger is not None:
        csv_logger.close()
    envs.close()
    # Return the raw module so callers can torch.save without worrying about
    # the DDP wrapper.
    return net.module if dctx.is_distributed else net
