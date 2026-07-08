"""Command-line interface: ``train``, ``eval``, ``play`` and ``record``."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from chaser_runner.agent import DQNAgent
from chaser_runner.config import load_config
from chaser_runner.env import ChaseEnv
from chaser_runner.logging_utils import NullLogger, TensorBoardLogger
from chaser_runner.trainer import Trainer, evaluate, make_agents, resolve_device


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.steps is not None:
        config.train.total_steps = args.steps

    if args.no_log:
        logger = NullLogger()
    else:
        run = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        logger = TensorBoardLogger(Path(args.logdir) / run)

    trainer = Trainer(config, device=args.device, logger=logger, model_dir=args.model_dir)
    trainer.load(args.resume_chaser, args.resume_runner)

    on_step = None
    if args.render:
        from chaser_runner.render import Renderer

        renderer = Renderer(trainer.env, fullscreen=args.fullscreen)

        def on_step(env, stats):
            if renderer.poll_should_quit():
                raise KeyboardInterrupt
            renderer.draw(stats)
            renderer.flip()

    try:
        trainer.train(on_step=on_step)
    except KeyboardInterrupt:
        print("\nInterrupted - models saved.")


def cmd_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = resolve_device(args.device)
    chaser, runner = make_agents(config, device)
    chaser.load(args.chaser)
    runner.load(args.runner)

    result = evaluate(config, chaser, runner, episodes=args.episodes)
    print(f"Episodes:            {result.episodes}")
    print(f"Capture rate:        {result.capture_rate:.1%}")
    print(f"Avg episode length:  {result.avg_length:.1f}")
    print(f"Avg reward (chaser): {result.avg_reward_chaser:.3f}")
    print(f"Avg reward (runner): {result.avg_reward_runner:.3f}")


def _load_pair(config, device, chaser_path, runner_path) -> tuple[DQNAgent, DQNAgent]:
    chaser, runner = make_agents(config, device)
    chaser.load(chaser_path)
    runner.load(runner_path)
    return chaser, runner


def cmd_play(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = resolve_device(args.device)
    env = ChaseEnv(config.env)
    chaser, runner = _load_pair(config, device, args.chaser, args.runner)

    from chaser_runner.render import Renderer

    renderer = Renderer(env, fullscreen=args.fullscreen)
    try:
        for _ in range(args.episodes):
            state = env.reset()
            done = False
            while not done:
                if renderer.poll_should_quit():
                    return
                act_ch = chaser.select_action(state, explore_eps=0.0)
                act_rn = runner.select_action(state, explore_eps=0.0)
                state, _, done, _ = env.step(act_ch, act_rn)
                renderer.draw({"mode": "showcase"})
                renderer.flip()
    finally:
        renderer.close()


def cmd_record(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio

    config = load_config(args.config)
    device = resolve_device(args.device)
    env = ChaseEnv(config.env)
    chaser, runner = _load_pair(config, device, args.chaser, args.runner)

    from chaser_runner.render import Renderer

    renderer = Renderer(env, headless=True)
    frames = []
    remaining = args.max_frames
    for _ in range(args.episodes):
        state = env.reset()
        done = False
        while not done and remaining > 0:
            act_ch = chaser.select_action(state, explore_eps=0.0)
            act_rn = runner.select_action(state, explore_eps=0.0)
            state, _, done, _ = env.step(act_ch, act_rn)
            renderer.draw()
            frames.append(renderer.frame())
            remaining -= 1
        if remaining <= 0:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames[:: args.stride], duration=1.0 / args.fps, loop=0)
    renderer.close()
    print(f"Wrote {len(frames[:: args.stride])} frames to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chaser-runner", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the chaser and runner via self-play.")
    _add_common(p_train)
    p_train.add_argument("--steps", type=int, default=None, help="Override total training steps.")
    p_train.add_argument("--model-dir", type=str, default="models")
    p_train.add_argument("--logdir", type=str, default="runs")
    p_train.add_argument("--run-name", type=str, default=None)
    p_train.add_argument("--no-log", action="store_true", help="Disable TensorBoard logging.")
    p_train.add_argument("--render", action="store_true", help="Show a live window while training.")
    p_train.add_argument("--fullscreen", action="store_true")
    p_train.add_argument("--resume-chaser", type=str, default=None)
    p_train.add_argument("--resume-runner", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate trained models over many episodes.")
    _add_common(p_eval)
    p_eval.add_argument("--chaser", type=str, default="models/chaser_q.pth")
    p_eval.add_argument("--runner", type=str, default="models/runner_q.pth")
    p_eval.add_argument("--episodes", type=int, default=50)
    p_eval.set_defaults(func=cmd_eval)

    p_play = sub.add_parser("play", help="Watch trained agents in a live Pygame window.")
    _add_common(p_play)
    p_play.add_argument("--chaser", type=str, default="models/chaser_q.pth")
    p_play.add_argument("--runner", type=str, default="models/runner_q.pth")
    p_play.add_argument("--episodes", type=int, default=5)
    p_play.add_argument("--fullscreen", action="store_true", help="Open in fullscreen.")
    p_play.set_defaults(func=cmd_play)

    p_rec = sub.add_parser("record", help="Render a headless GIF of trained agents.")
    _add_common(p_rec)
    p_rec.add_argument("--chaser", type=str, default="models/chaser_q.pth")
    p_rec.add_argument("--runner", type=str, default="models/runner_q.pth")
    p_rec.add_argument("--out", type=str, default="assets/demo.gif")
    p_rec.add_argument("--episodes", type=int, default=3)
    p_rec.add_argument("--max-frames", type=int, default=1500)
    p_rec.add_argument("--stride", type=int, default=2, help="Keep every Nth frame.")
    p_rec.add_argument("--fps", type=int, default=30)
    p_rec.set_defaults(func=cmd_record)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
