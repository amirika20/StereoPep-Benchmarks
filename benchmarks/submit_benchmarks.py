#!/usr/bin/env python3
"""
Submit SLURM array jobs for DeepRT benchmarks.

Each benchmark is submitted as a SLURM array job where task index == seed,
so results land in benchmarks/results_<name>_seed<N>.json.

Usage
-----
  # Submit all benchmarks, seeds 0-4, on Kempner
  python benchmarks/submit_benchmarks.py

  # Submit only two benchmarks with 10 seeds
  python benchmarks/submit_benchmarks.py --benchmarks morgan_fp_mlp pretrained_gin --seeds 0-9

  # Dry run (print sbatch scripts, don't submit)
  python benchmarks/submit_benchmarks.py --dry-run

  # List available benchmarks
  python benchmarks/submit_benchmarks.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── Benchmark registry ────────────────────────────────────────────────────────
# Each entry specifies resource overrides relative to the cluster defaults.
# Omit a key to use the cluster default.

BENCHMARKS: dict[str, dict] = {
    "morgan_fp_mlp": {
        "script":  "benchmarks/morgan_fp_mlp.py",
        "time":    "0-04:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "transformer_scratch": {
        "script":  "benchmarks/transformer_scratch.py",
        "time":    "0-06:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "deeplc": {
        "script":  "benchmarks/deeplc.py",
        "time":    "0-06:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "deeprt_capsnet": {
        "script":  "benchmarks/deeprt_capsnet.py",
        "time":    "0-08:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "pretrained_gin": {
        "script":  "benchmarks/pretrained_gin.py",
        "time":    "0-06:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "gin_scratch": {
        "script":  "benchmarks/gin_scratch.py",
        "time":    "0-06:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "egnn_3d": {
        "script":  "benchmarks/egnn_3d.py",
        "time":    "0-08:00",
        "mem":     "32G",
        "gpus":    1,
    },
    "esm3_sm": {
        "script":  "benchmarks/esm3_embedding.py",
        "time":    "0-12:00",
        "mem":     "64G",
        "gpus":    1,
        "extra":   ["--model", "esm3_sm"],
    },
    "esmc_300m": {
        "script":  "benchmarks/esm3_embedding.py",
        "time":    "0-12:00",
        "mem":     "64G",
        "gpus":    1,
        "extra":   ["--model", "esmc_300m"],
    },
    "esmc_600m": {
        "script":  "benchmarks/esm3_embedding.py",
        "time":    "0-16:00",
        "mem":     "80G",   # larger model
        "gpus":    1,
        "extra":   ["--model", "esmc_600m"],
    },
}

# Root of the repository (one level up from this file)
REPO_ROOT = Path(__file__).parent.parent
SUBMIT_SH  = REPO_ROOT / "submit.sh"


# ── Helpers ───────────────────────────────────────────────────────────────────

def seed_range_str(seeds: str) -> str:
    """Normalise seeds to 'A-B' or 'A B C' as needed by submit.sh."""
    return seeds


def submit_one(
    name: str,
    cfg: dict,
    seeds: str,
    epochs: int,
    dry_run: bool,
    extra_args: list[str],
) -> None:
    cmd = [
        str(SUBMIT_SH),
        "--benchmark",
        "--seeds",    seeds,
        "--time",     cfg.get("time", ""),
        "--mem",      cfg.get("mem",  ""),
        "--gpus",     str(cfg.get("gpus", 1)),
        "--name",     name,
    ]
    if dry_run:
        cmd.append("--dry-run")

    # Remove empty flags (e.g. time/mem not set)
    cleaned: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] in ("--time", "--mem") and (i + 1 >= len(cmd) or cmd[i + 1] == ""):
            i += 2  # skip flag + empty value
        else:
            cleaned.append(cmd[i])
            i += 1

    cleaned += ["--", cfg["script"], "--epochs", str(epochs)]
    cleaned += cfg.get("extra", [])
    cleaned += extra_args

    print(f"\n{'─' * 60}")
    print(f"Benchmark : {name}")
    print(f"Script    : {cfg['script']}")
    print(f"Seeds     : {seeds}")
    print(f"Epochs    : {epochs}")
    print(f"Command   : {' '.join(cleaned)}")
    print(f"{'─' * 60}")

    result = subprocess.run(cleaned, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] submit.sh failed for {name} (exit {result.returncode})", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit SLURM array jobs for all DeepRT benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmarks", nargs="+", metavar="NAME",
        default=list(BENCHMARKS),
        help="Benchmarks to submit (default: all). Use --list to see names.",
    )
    parser.add_argument(
        "--seeds", default="0-9",
        help="Seed range or list, e.g. '0-9' or '0 1 2 3 4' (default: 0-9)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Max training epochs passed to each benchmark (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmark names and exit",
    )
    args, extra = parser.parse_known_args()

    if args.list:
        print("Available benchmarks:")
        for name, cfg in BENCHMARKS.items():
            print(f"  {name:<22}  {cfg['script']}")
        return

    unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
    if unknown:
        parser.error(f"Unknown benchmark(s): {', '.join(unknown)}. "
                     f"Use --list to see valid names.")

    if not SUBMIT_SH.exists():
        parser.error(f"submit.sh not found at {SUBMIT_SH}")

    print(f"Submitting {len(args.benchmarks)} benchmark(s) on Kempner")
    print(f"Seeds: {args.seeds}  |  Epochs: {args.epochs}")
    if args.dry_run:
        print("(dry run)")

    for name in args.benchmarks:
        submit_one(
            name=name,
            cfg=BENCHMARKS[name],
            seeds=args.seeds,
            epochs=args.epochs,
            dry_run=args.dry_run,
            extra_args=extra,
        )

    print(f"\nDone. Results will be written to benchmarks/results_<name>_seed<N>.json")


if __name__ == "__main__":
    main()
