"""Minimal Gymnasium env wrapping PyBoy + Pokemon Red.

v0: no reward (always returns 0), no termination logic beyond a step
budget. Goal is to confirm the env loop runs end-to-end with a random
policy. Reward function and termination conditions land in later
modules.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

from pokerl.env import ram_map as rm
from pokerl.env.rewards import Reward, RewardV0_1

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROM = ROOT / "roms" / "pokemon_red.gb"
DEFAULT_STATE = ROOT / "states" / "post_intro.state"

# Action index -> PyBoy button name. "noop" advances frames without input.
ACTIONS: tuple[str, ...] = ("noop", "a", "b", "up", "down", "left", "right")

# Game ticks per agent step. 24 ≈ 0.4s of game time. Standard practice for
# Game Boy RL — gives buttons time to register and animations to play out.
FRAMES_PER_STEP = 24

# How long to hold a button down (in frames) before releasing. Pokemon Red's
# movement check requires the direction held for at least ~8 frames; the
# default pyboy.button() press of 1 frame is far too short to register
# movement commands. We hold for half the step and let the rest of the step
# play out animations.
BUTTON_HOLD_FRAMES = 12

# Observation: downsampled grayscale screen.
SCREEN_H = 72   # 144 / 2
SCREEN_W = 80   # 160 / 2

# Pokemon Red (MBC3) has 32KB of battery-backed save RAM. We supply an
# in-memory pre-zeroed buffer of this exact size as PyBoy's ram_file so
# it doesn't auto-create a `<rom>.ram` companion on disk. Each env
# subprocess gets its own buffer — no NFS contention at cluster startup.
CART_RAM_BYTES = 32768


class PokemonRedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | Path = DEFAULT_ROM,
        state_path: str | Path = DEFAULT_STATE,
        headless: bool = True,
        max_steps: int = 4096,
        reward: Reward | None = None,
        frame_skip: int = 1,
    ) -> None:
        super().__init__()
        self.rom_path = Path(rom_path)
        self.state_path = Path(state_path)
        self.max_steps = max_steps
        self.reward_fn: Reward = reward if reward is not None else RewardV0_1()
        self.frame_skip = max(1, int(frame_skip))

        if not self.rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {self.rom_path}")
        if not self.state_path.exists():
            raise FileNotFoundError(
                f"Save state not found: {self.state_path}. "
                "Run: uv run python -m pokerl.scripts.make_save_state"
            )

        # ram_file=pre-zeroed BytesIO keeps PyBoy from auto-creating a
        # `<rom>.ram` companion on disk. With N PyBoy subprocesses racing
        # for the same NFS file on cluster startup, partial reads cause
        # PyBoyAssertException ("No data") and crash rank 0 — bringing the
        # whole DDP group down via NCCL timeout. Each env gets its own
        # in-memory save-RAM instead. Must be pre-sized to the cartridge's
        # save RAM size or PyBoy's load_ram raises "No data".
        self.pyboy = PyBoy(
            str(self.rom_path),
            window="null" if headless else "SDL2",
            ram_file=io.BytesIO(b"\x00" * CART_RAM_BYTES),
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(SCREEN_H, SCREEN_W), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._steps = 0

    # --- gym.Env interface ---

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick()
        self.reward_fn.reset(self.pyboy.memory)
        self._steps = 0
        return self._obs(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        name = ACTIONS[int(action)]
        running = True
        for _ in range(self.frame_skip):
            if name != "noop":
                self.pyboy.button_press(name)
            for frame in range(FRAMES_PER_STEP):
                if name != "noop" and frame == BUTTON_HOLD_FRAMES:
                    self.pyboy.button_release(name)
                running = self.pyboy.tick()
                if not running:
                    break
            if name != "noop":
                self.pyboy.button_release(name)
            if not running:
                break

        self._steps += 1
        obs = self._obs()
        reward = self.reward_fn.compute(self.pyboy.memory)
        terminated = False
        truncated = self._steps >= self.max_steps
        info: dict[str, Any] = {"step": self._steps, **self._info()}
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._screen_rgb()

    def close(self) -> None:
        if self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None  # type: ignore[assignment]

    # --- internals ---

    def _screen_rgb(self) -> np.ndarray:
        """Return the current frame as (H, W, 3) uint8 RGB."""
        arr = self.pyboy.screen.ndarray
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    def unique_tiles_visited(self) -> int:
        """Expose reward-fn exploration stats across vector wrappers (incl. async)."""
        fn = getattr(self, "reward_fn", None)
        return int(getattr(fn, "unique_tiles_visited", 0)) if fn is not None else 0

    def _obs(self) -> np.ndarray:
        """Downsampled grayscale screen, (SCREEN_H, SCREEN_W) uint8."""
        rgb = self._screen_rgb()
        gray = rgb.mean(axis=-1).astype(np.uint8)
        return gray[::2, ::2]

    def _info(self) -> dict[str, Any]:
        """Side-channel state for the RND learner (NOT part of the policy obs):
          - 'progress': the 4 hidden story-progress bits (parcel/pokedex/brock/
            pokeballs) that pixels can't show, so curiosity is context-aware.
          - 'in_battle': battle flag so the PPO loop can zero intrinsic reward
            during battles (their RNG is unlearnable noisy-TV).
        Threading these through info (rather than a Dict obs space) keeps
        FrameStack + AsyncVectorEnv untouched."""
        mem = self.pyboy.memory
        return {
            "progress": np.asarray(rm.progress_bits(mem), dtype=np.float32),
            "in_battle": int(rm.in_battle(mem)),
        }
