"""One-shot health card for a training run — the thing you'd otherwise SSH and
tail metrics.csv for. Pure CSV arithmetic over runs/<run>/metrics.csv; no deps.

  uv run python -m pokerl.scripts.status elsa_genoa_cold_h32k            # local runs/
  uv run python -m pokerl.scripts.status elsa_genoa_cold_h32k --remote   # ssh-cat from ELSA

Reports: progress + ETA, sps (now vs median), return trend, entropy vs max,
RND novelty decay, episodes/tiles, rollout/update split, last-checkpoint age.
Flags the project's documented failure modes (entropy collapse, novelty->0,
return plateau, throughput drop) but does NOT act on them — alert, don't kill.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = "guenthc1@elsa.hpc.tcnj.edu"
REPO = "/scratch/guenthc1/pokemon-rl"
N_ACTIONS = 7              # max policy entropy is log(N_ACTIONS)
H_MAX = math.log(N_ACTIONS)


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _ssh(cmd: str) -> str:
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, cmd],
        text=True, capture_output=True,
    )
    return out.stdout


def _fmt_dt(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # nan-safe
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load(run: str, remote: bool) -> tuple[list[dict], dict, str]:
    """Return (csv_rows, meta_dict, checkpoint_age_str)."""
    if remote:
        text = _ssh(f"cat {REPO}/runs/{run}/metrics.csv 2>/dev/null")
        meta_raw = _ssh(f"cat {REPO}/runs/{run}/meta.json 2>/dev/null")
        ck = _ssh(
            f"ls -t {REPO}/runs/{run}/checkpoints/iter_*.pt 2>/dev/null | head -1"
        ).strip()
        ck_age = ""
        if ck:
            epoch = _ssh(f"date +%s").strip()
            mtime = _ssh(f"stat -c %Y '{ck}' 2>/dev/null").strip()
            if epoch.isdigit() and mtime.isdigit():
                ck_age = f"{_fmt_dt(int(epoch) - int(mtime))} ago ({Path(ck).name})"
    else:
        run_dir = ROOT / "runs" / run
        text = (run_dir / "metrics.csv").read_text() if (run_dir / "metrics.csv").exists() else ""
        mp = run_dir / "meta.json"
        meta_raw = mp.read_text() if mp.exists() else ""
        cks = sorted((run_dir / "checkpoints").glob("iter_*.pt")) if (run_dir / "checkpoints").exists() else []
        ck_age = cks[-1].name if cks else ""
    rows = list(csv.DictReader(io.StringIO(text))) if text.strip() else []
    meta = json.loads(meta_raw) if meta_raw.strip() else {}
    return rows, meta, ck_age


def card(run: str, rows: list[dict], meta: dict, ck_age: str, window: int) -> str:
    if not rows:
        return f"[{run}] no metrics yet (run hasn't logged its first iter, or wrong run_name)."
    last = rows[-1]
    win = rows[-window:] if len(rows) >= 2 else rows

    it = int(_f(last, "iteration"))
    step = int(_f(last, "global_step"))
    sps_now = _f(last, "samples_per_second")
    sps_med = statistics.median(_f(r, "samples_per_second") for r in rows)
    H = _f(last, "entropy")
    nov_now = _f(last, "mean_raw_novelty")
    nov_first = _f(win[0], "mean_raw_novelty")
    ret_now = _f(last, "mean_return_last20")
    ret_first = _f(win[0], "mean_return_last20")
    eps = int(_f(last, "episodes_completed"))
    tiles = int(_f(last, "unique_tiles_total"))
    roll = _f(last, "rollout_seconds")
    upd = _f(last, "update_seconds")
    total = int(meta.get("total_timesteps", 0))

    lines = [f"=== {run} ==="]
    if meta:
        dirty = " (DIRTY)" if meta.get("dirty") else ""
        lines.append(
            f"  {meta.get('reward_class','?')}  seed {meta.get('seed','?')}  "
            f"threads {meta.get('torch_threads','?')}  commit {str(meta.get('commit',''))[:8]}{dirty}"
        )
    # progress + ETA
    if total:
        pct = 100.0 * step / total
        eta = (total - step) / sps_now if sps_now > 0 else float("nan")
        lines.append(f"  progress : iter {it}  step {step:,}/{total:,} ({pct:.1f}%)  ETA {_fmt_dt(eta)}")
    else:
        lines.append(f"  progress : iter {it}  step {step:,}")
    # throughput
    sps_flag = "  <- DROP vs median" if sps_now < 0.7 * sps_med and sps_med > 0 else ""
    lines.append(f"  sps      : {sps_now:.0f}  (median {sps_med:.0f}){sps_flag}   roll {roll:.1f}s / upd {upd:.1f}s")
    # learning signals
    d_ret = ret_now - ret_first
    lines.append(f"  return   : {ret_now:+.3f}  (Δ{d_ret:+.3f} over last {len(win)} logs)   episodes {eps}  tiles {tiles}")
    lines.append(f"  entropy  : {H:.3f} / {H_MAX:.3f} max  ({100*H/H_MAX:.0f}%)")
    d_nov = nov_now - nov_first
    lines.append(f"  novelty  : {nov_now:.4f}  (Δ{d_nov:+.4f})   rnd_loss {_f(last,'rnd_loss'):.4f}")
    if ck_age:
        lines.append(f"  last ckpt: {ck_age}")

    # health flags — alert only. The entropy rule is probe-aware: low entropy is
    # the INTENDED state during a diagnostic probe, so it only fires when return
    # is ALSO flat (i.e. committed to a degenerate strategy, not just exploring).
    warns = []
    if H < 0.4 and d_ret <= 0.001:
        warns.append("entropy collapsed AND return flat -> possible degenerate policy (watch a checkpoint)")
    if nov_now < 1e-3 and abs(d_nov) < 1e-3:
        warns.append("novelty ~0 and not moving -> curiosity exhausted / exploration stalled")
    if abs(d_ret) < 1e-3 and len(win) >= window:
        warns.append(f"return flat over last {len(win)} logs -> plateau")
    if sps_med > 0 and sps_now < 0.7 * sps_med:
        warns.append("throughput dropped >30% vs median -> job may be contended/wedged")
    if eps == 0 and it > 50:
        warns.append("no episodes completed yet -> long horizon or stuck (expected early at h32k)")

    if warns:
        lines.append("  health   : WARN")
        for w in warns:
            lines.append(f"     - {w}")
    else:
        lines.append("  health   : OK")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run_name (dir under runs/)")
    ap.add_argument("--remote", action="store_true", help="ssh-cat metrics from ELSA instead of local runs/")
    ap.add_argument("-n", "--window", type=int, default=20, help="how many recent logs to measure trends over")
    ap.add_argument("--check", action="store_true", help="exit 1 if any health warning fires (for scripts)")
    args = ap.parse_args()

    rows, meta, ck_age = load(args.run, args.remote)
    text = card(args.run, rows, meta, ck_age, args.window)
    print(text)
    if args.check:
        raise SystemExit(1 if "health   : WARN" in text else 0)


if __name__ == "__main__":
    main()
