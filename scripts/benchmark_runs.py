"""Compare untrained, latest, and selected policies on held-out benchmark episodes.

Usage: python scripts/benchmark_runs.py runs/seed0 runs/seed1 --episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chaser_runner.agent import FrozenPolicy
from chaser_runner.config import Config
from chaser_runner.evaluation import BaselinePolicy, benchmark, evaluate
from chaser_runner.persistence import atomic_json, isolated_rng, read_checkpoint
from chaser_runner.trainer import make_agents


def report_run(run: Path, episodes: int, seed: int) -> dict:
    latest = read_checkpoint(run / "policies" / "latest.pt")
    config = Config.from_dict(latest["config"])
    torch.set_num_threads(config.train.torch_threads)
    result = {
        "run": str(run),
        "training_seed": config.train.seed,
        "training_steps": latest["step"],
        "config": config.to_dict(),
    }
    with isolated_rng(config.train.seed):
        chaser, runner = make_agents(config, "cpu")
    result["untrained"] = benchmark(config, chaser, runner, episodes=episodes, seed=seed)
    result["latest"] = benchmark(
        config,
        FrozenPolicy(config.agent, latest["policies"]["chaser"]),
        FrozenPolicy(config.agent, latest["policies"]["runner"]),
        episodes=episodes,
        seed=seed,
    )
    best_ch = read_checkpoint(run / "policies" / "best_chaser.pt")
    best_rn = read_checkpoint(run / "policies" / "best_runner.pt")
    result["selected_steps"] = {"chaser": best_ch["step"], "runner": best_rn["step"]}
    result["selected"] = benchmark(
        config,
        FrozenPolicy(config.agent, best_ch["policies"]["chaser"]),
        FrozenPolicy(config.agent, best_rn["policies"]["runner"]),
        episodes=episodes,
        seed=seed,
    )
    result["scripted_pair"] = evaluate(
        config,
        BaselinePolicy("chaser", config.env),
        BaselinePolicy("runner", config.env),
        episodes=episodes,
        seed=seed,
    ).to_dict()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=Path("evaluations/learning_report.json"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    runs = []
    for run in args.runs:
        result = report_run(run, args.episodes, args.seed)
        runs.append(result)
        atomic_json({"episodes": args.episodes, "seed": args.seed, "runs": runs}, args.out)
        print(
            f"Training seed {result['training_seed']}, selected steps {result['selected_steps']}",
            flush=True,
        )
        for category in ("untrained", "latest", "selected"):
            ch = result[category]["chaser_vs_scripted"]
            rn = result[category]["scripted_vs_runner"]
            rnd = result[category]["chaser_vs_random"]
            print(
                f"  {category}: chaser capture scripted={ch['capture_rate']:.0%}, "
                f"random={rnd['capture_rate']:.0%}; runner length={rn['avg_length']:.1f} ticks, "
                f"survival={rn['survival_rate']:.0%}",
                flush=True,
            )
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
