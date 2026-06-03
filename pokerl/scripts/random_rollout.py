"""Run a random policy in PokemonRedEnv to verify the env loop works.

End-to-end test: env constructs, resets, steps with random actions,
returns sane shapes. No learning. Useful first sanity check after any
env change.
"""
from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pokerl.env.pokemon_red_env import PokemonRedEnv

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "notes"
N_STEPS = 200


def main() -> None:
    env = PokemonRedEnv(headless=True, max_steps=N_STEPS + 1)
    rng = np.random.default_rng(0)

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)
    assert obs.dtype == np.uint8

    start = time.time()
    frames_for_gif = [env.render()]
    for i in range(N_STEPS):
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 25 == 0:
            frames_for_gif.append(env.render())
        if terminated or truncated:
            break
    elapsed = time.time() - start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = OUT_DIR / "random_rollout.gif"
    iio.imwrite(gif_path, np.array(frames_for_gif), duration=200, loop=0)
    final_png = OUT_DIR / "random_rollout_final.png"
    iio.imwrite(final_png, env.render())

    env.close()

    print(f"Ran {N_STEPS} random steps in {elapsed:.2f}s "
          f"({N_STEPS / elapsed:.1f} steps/s)")
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Wrote gif:   {gif_path} ({len(frames_for_gif)} frames)")
    print(f"Wrote final: {final_png}")


if __name__ == "__main__":
    main()
