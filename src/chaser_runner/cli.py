"""Train, resume, benchmark, watch, or record explicit saved policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from chaser_runner.agent import FrozenPolicy
from chaser_runner.config import Config, load_config
from chaser_runner.env import ChaseEnv
from chaser_runner.evaluation import BaselinePolicy, benchmark, evaluate
from chaser_runner.logging_utils import TensorBoardLogger
from chaser_runner.persistence import atomic_json, read_checkpoint, seed_all
from chaser_runner.trainer import Trainer, resolve_device


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]


def cmd_train(args) -> None:
    if args.resume:
        if args.config or args.run_name or args.warm_start_chaser or args.warm_start_runner:
            raise ValueError(
                "Resume uses the saved config and run directory; do not combine with "
                "--config, --run-name or warm starts"
            )
        trainer = Trainer.resume(args.resume, device=args.device)
    else:
        config = load_config(args.config)
        if args.steps is not None:
            config.train.total_steps = args.steps
        config.__post_init__()
        run_dir = Path(args.logdir) / (args.run_name or _stamp())
        trainer = Trainer(config, device=args.device, run_dir=run_dir)
        for role in ("chaser", "runner"):
            path = getattr(args, f"warm_start_{role}")
            if path:
                policy = _load_policy(path, role, config, trainer.device, warm_start=True)
                if not isinstance(policy, FrozenPolicy):
                    raise ValueError("Warm starts require a learned policy file")
                agent = trainer.agents[role]
                net = agent.actor if config.agent.algorithm == "td3" else agent.q
                net.load_state_dict(policy.net.state_dict())
                target = agent.actor_target if config.agent.algorithm == "td3" else agent.q_target
                target.load_state_dict(net.state_dict())
        if args.warm_start_chaser or args.warm_start_runner:
            # Rebuild the initial historical opponent after loading warm-start policies.
            trainer.pool.clear()
            trainer._snapshot()
            trainer._start_episode()
            atomic_json(
                {"chaser": args.warm_start_chaser, "runner": args.warm_start_runner},
                trainer.run_dir / "warm_start.json",
            )
    if args.steps is not None:
        trainer.config.train.total_steps = args.steps
    trainer.config.__post_init__()
    if not args.no_log:
        trainer.logger = TensorBoardLogger(
            trainer.run_dir / "tensorboard",
            purge_step=trainer.global_step + 1 if args.resume else None,
        )
    print(f"Run directory: {trainer.run_dir}", flush=True)
    renderer = None
    on_step = None
    try:
        if args.render:
            from chaser_runner.render import Renderer

            renderer = Renderer(trainer.env, fullscreen=args.fullscreen)

            def on_step(env, stats):
                if renderer.poll_should_quit():
                    trainer.request_stop()
                renderer.draw(stats)
                renderer.flip()

        trainer.train(on_step=on_step)
    finally:
        if renderer is not None:
            renderer.close()
    print(
        f"Saved step {trainer.global_step}. Resume with: chaser-runner train "
        f'--resume "{trainer.run_dir}" --steps <total-target>'
    )


def _read_source(path) -> dict:
    path = Path(path)
    if path.is_dir() and (path / "policies" / "latest.pt").exists():
        path = path / "policies" / "latest.pt"
    return read_checkpoint(path)


def _load_policy(source, role, config, device, *, warm_start=False):
    if source in {"scripted", "random", "stationary"}:
        return BaselinePolicy(role, config.env, source)
    if Path(source).suffix == ".pth":
        state = torch.load(source, map_location="cpu", weights_only=True)
    else:
        data = _read_source(source)
        saved_config = Config.from_dict(data["config"])
        if warm_start:
            if (saved_config.agent.algorithm, saved_config.agent.hidden) != (
                config.agent.algorithm,
                config.agent.hidden,
            ):
                raise ValueError("Warm-start network architecture must match the saved policy")
        elif saved_config.env != config.env or saved_config.agent != config.agent:
            raise ValueError(
                "Policy configurations differ; use matching environment and agent configs"
            )
        if data["kind"] == "policy":
            state = data["policies"][role]
        else:
            key = "actor" if config.agent.algorithm == "td3" else "q"
            state = data["agents"][role]["modules"][key]
    return FrozenPolicy(config.agent, state, device)


def _load_matchup(args):
    sources = {"chaser": args.chaser or args.checkpoint, "runner": args.runner or args.checkpoint}
    if any(source is None for source in sources.values()):
        raise ValueError("Supply --checkpoint RUN_OR_BUNDLE, or both --chaser and --runner")
    config = None
    for source in sources.values():
        if source not in {"scripted", "random", "stationary"} and Path(source).suffix != ".pth":
            config = Config.from_dict(_read_source(source)["config"])
            break
    if args.config:
        explicit = load_config(args.config)
        if config and (explicit.env != config.env or explicit.agent != config.agent):
            raise ValueError("--config disagrees with the saved policy configuration")
        config = explicit
    if config is None:
        if any(Path(source).suffix == ".pth" for source in sources.values()) and not args.config:
            raise ValueError(
                "Raw .pth weights require --config with their architecture and physics"
            )
        config = load_config(args.config)
    torch.set_num_threads(config.train.torch_threads)
    device = resolve_device(args.device)
    chaser = _load_policy(sources["chaser"], "chaser", config, device)
    runner = _load_policy(sources["runner"], "runner", config, device)
    seed_all(args.seed)
    return config, chaser, runner, sources


def cmd_eval(args) -> None:
    config, chaser, runner, sources = _load_matchup(args)
    if args.benchmark:
        results = benchmark(config, chaser, runner, episodes=args.episodes, seed=args.seed)
    else:
        results = {
            "paired": evaluate(
                config, chaser, runner, episodes=args.episodes, seed=args.seed
            ).to_dict()
        }
    for name, result in results.items():
        print(
            f"{name}: capture={result['capture_rate']:.1%} "
            f"(95% CI {result['capture_ci_low']:.1%}-{result['capture_ci_high']:.1%}), "
            f"survival={result['survival_rate']:.1%}, "
            f"length={result['avg_length']:.1f} ticks, distance={result['avg_distance']:.1f}"
        )
    output = args.out
    if output is None:
        base = Path(args.checkpoint) if args.checkpoint else None
        output = (base / "evaluations") if base and base.is_dir() else Path("evaluations")
        output = output / f"manual_{_stamp()}.json"
    atomic_json(
        {
            "seed": args.seed,
            "episodes": args.episodes,
            "sources": sources,
            "config": config.to_dict(),
            "matchups": results,
        },
        output,
    )
    print(f"Evaluation saved: {output}")


def cmd_play(args) -> None:
    config, chaser, runner, _ = _load_matchup(args)
    env = ChaseEnv(config.env)
    from chaser_runner.render import Renderer

    renderer = Renderer(env, fullscreen=args.fullscreen)
    try:
        for ep in range(args.episodes):
            state = env.reset(seed=args.seed + ep)
            done = False
            while not done:
                if renderer.poll_should_quit():
                    return
                state, _, done, _ = env.step(
                    chaser.select_action(state), runner.select_action(state)
                )
                renderer.draw({"mode": "showcase"})
                renderer.flip()
    finally:
        renderer.close()


def cmd_record(args) -> None:
    import imageio.v2 as imageio

    config, chaser, runner, _ = _load_matchup(args)
    env = ChaseEnv(config.env)
    from chaser_runner.render import Renderer

    renderer = Renderer(env, headless=True)
    frames = []
    remaining = args.max_frames
    try:
        for ep in range(args.episodes):
            state = env.reset(seed=args.seed + ep)
            done = False
            while not done and remaining:
                state, _, done, _ = env.step(
                    chaser.select_action(state), runner.select_action(state)
                )
                renderer.draw()
                frames.append(renderer.frame())
                remaining -= 1
            if not remaining:
                break
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = args.fps or config.env.tickrate / config.env.action_repeat
        imageio.mimsave(out, frames[:: args.stride], duration=1000 * args.stride / fps, loop=0)
    finally:
        renderer.close()
    print(f"Wrote {len(frames[:: args.stride])} frames to {out}")


def _positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _common(parser):
    parser.add_argument("--config", help="YAML config; saved bundles already contain their config.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")


def _policies(parser):
    _common(parser)
    parser.add_argument("--checkpoint", help="Run directory, policy bundle, or full checkpoint")
    parser.add_argument(
        "--chaser", help="Override chaser: bundle, raw .pth, scripted, random, stationary"
    )
    parser.add_argument(
        "--runner", help="Override runner: bundle, raw .pth, scripted, random, stationary"
    )
    parser.add_argument("--seed", type=int, default=20_000)


def build_parser():
    parser = argparse.ArgumentParser(prog="chaser-runner", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    _common(train)
    train.add_argument(
        "--steps", type=_positive, help="Total target policy decisions, including resumed progress"
    )
    train.add_argument("--logdir", default="runs", help="Parent directory for new runs")
    train.add_argument("--run-name", help="Unique name for a new run")
    train.add_argument("--resume", help="Continue a run directory or its latest full checkpoint")
    train.add_argument("--warm-start-chaser", "--resume-chaser", dest="warm_start_chaser")
    train.add_argument("--warm-start-runner", "--resume-runner", dest="warm_start_runner")
    train.add_argument(
        "--no-log", action="store_true", help="Disable TensorBoard; keep run artifacts"
    )
    train.add_argument("--render", action="store_true")
    train.add_argument("--fullscreen", action="store_true")
    train.set_defaults(func=cmd_train)
    ev = sub.add_parser("eval")
    _policies(ev)
    ev.add_argument("--episodes", type=_positive, default=50)
    ev.add_argument(
        "--benchmark", action="store_true", help="Also evaluate fixed reference opponents"
    )
    ev.add_argument("--out", help="JSON output path (generated automatically when omitted)")
    ev.set_defaults(func=cmd_eval)
    play = sub.add_parser("play")
    _policies(play)
    play.add_argument("--episodes", type=_positive, default=5)
    play.add_argument("--fullscreen", action="store_true")
    play.set_defaults(func=cmd_play)
    record = sub.add_parser("record")
    _policies(record)
    record.add_argument("--episodes", type=_positive, default=3)
    record.add_argument("--out", default="assets/demo.gif")
    record.add_argument("--max-frames", type=_positive, default=1500)
    record.add_argument("--stride", type=_positive, default=1)
    record.add_argument("--fps", type=_positive, help="Override policy-decision playback FPS")
    record.set_defaults(func=cmd_record)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "seed") and args.seed < 0:
        parser.error("--seed must be nonnegative")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
