# Chaser-Runner

> Adversarial self-play reinforcement learning: a **chaser** learns to pursue and a **runner** learns to evade, on a wrap-around 2D world — both trained from scratch, at the same time, against each other.

<p align="center">
  <img src="assets/demo.gif" alt="Trained chaser (red) pursuing runner (green) on a toroidal world" width="480">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#command-reference">Commands</a> ·
  <a href="#project-layout">Layout</a>
</p>

---

## What is this?

Two agents share the same continuous-control physics but have opposite goals. The
chaser is rewarded for closing the distance (and gets a bonus for a capture); the
runner is rewarded for opening it. Because they train **simultaneously**, each one
is a moving target for the other — a small self-play arms race that produces
surprisingly natural chasing and juking behaviour.

- **Continuous actions.** Each agent outputs a 2D acceleration in `[-1, 1]²`.
- **Sampling-based DQN.** Instead of a discrete action head, a `Q(s, a)` network
  scores many *sampled* candidate actions and the best one is played. This keeps
  the DQN machinery while handling a continuous action space.
- **Toroidal world.** The arena wraps around at every edge, so the shortest path
  between two agents may go *through* a wall.
- **Runs, trains, and showcases.** Train with live TensorBoard curves, evaluate
  over many episodes, watch a fullscreen live match, or record a GIF headlessly.

## Quickstart

```bash
# 1. Install (editable, with dev tooling)
pip install torch            # or the CUDA build for your platform
pip install -e ".[dev]"

# 2. Train from scratch (Ctrl+C any time — models are saved on exit)
chaser-runner train --steps 200000

# 3. Watch the trained agents in a live window
chaser-runner play --fullscreen

# 4. See how good they are
chaser-runner eval --episodes 50
```

> Trained weights are not checked into the repo — run `train` first (it writes
> `models/chaser_q.pth` and `models/runner_q.pth`), which `play`/`eval` then load.

> **GPU:** training auto-detects CUDA. Pass `--device cpu` to force CPU.

## Command reference

All commands are subcommands of `chaser-runner` (see `chaser-runner <cmd> -h`).

| Command  | What it does                                              | Key flags |
| -------- | --------------------------------------------------------- | --------- |
| `train`  | Self-play training loop with TensorBoard logging          | `--steps`, `--config`, `--render`, `--fullscreen`, `--resume-chaser/-runner`, `--no-log` |
| `eval`   | Roll out greedy policies, report capture rate & rewards   | `--episodes`, `--chaser`, `--runner` |
| `play`   | Watch trained agents in a live Pygame window              | `--fullscreen`, `--episodes` |
| `record` | Render a GIF **headlessly** (no display needed)           | `--out`, `--episodes`, `--fps`, `--stride` |

### Train, retrain, and experiment

Everything is driven by [`configs/default.yaml`](configs/default.yaml). To try a
variation, copy it, edit the physics or learner, and point `train` at it:

```bash
cp configs/default.yaml configs/fast_runner.yaml
# edit e.g. env.max_speed, agent.lr, train.eps_decay_steps ...
chaser-runner train --config configs/fast_runner.yaml --run-name fast_runner
```

**Continue training** from existing weights instead of starting over:

```bash
chaser-runner train --resume-chaser models/chaser_q.pth \
                     --resume-runner models/runner_q.pth --steps 50000
```

### Watch the training curves

```bash
tensorboard --logdir runs
```

You get live `reward`, `loss`, `distance`, `episode length`, and `capture` curves —
screenshot these for your write-up.

### Record a demo GIF

```bash
chaser-runner record --out assets/demo.gif --episodes 3
```

This uses SDL's dummy video driver, so it works over SSH, in Docker, and in CI.

## How it works

```
             ┌──────────────┐   action_ch    ┌──────────────┐
   state ───►│ Chaser agent │───────────────►│              │
     ▲       └──────────────┘                 │   ChaseEnv   │──► reward_ch, reward_rn
     │       ┌──────────────┐   action_rn     │  (toroidal)  │      next state, done
     └───────│ Runner agent │───────────────►│              │
             └──────────────┘                 └──────────────┘
```

**State** (6-dim, relative & normalized): `[rel_disp(2), chaser_vel(2),
runner_vel(2)]`, where `rel_disp` is the shortest wrap-around vector from chaser to
runner. On a homogeneous toroidal world only *relative* geometry matters, so
absolute position is deliberately dropped and every component is scaled to roughly
`[-1, 1]` — this is what makes the task learnable.

**Reward** is zero-sum on the shaped distance term: the chaser earns
`(prev_dist − dist) / world_size` each step, the runner earns its negation. Capture
adds `+1` for the chaser and `−1` for the runner and ends the episode; reaching the
step limit ends it as a runner "win".

**Game balance.** The chaser is slightly faster (`env.chaser_advantage`, default
`1.2`). Equal speed on a torus makes a competent runner *uncatchable*, so the
chaser needs an edge to ever learn to win.

**Learning.** Each agent is an independent [`DQNAgent`](src/chaser_runner/agent.py)
with its own replay buffer and target network. Since actions are continuous, both
the acting policy and the Bellman `max` are approximated by scoring uniformly
sampled candidate actions with `Q(s, a)` and taking the best.

## Project layout

```
src/chaser_runner/
├── config.py         # typed dataclass config, loadable from YAML
├── env.py            # ChaseEnv — pure toroidal physics (no rendering)
├── networks.py       # QNetwork: Q(s, a) → scalar value
├── replay.py         # experience replay buffer
├── agent.py          # DQNAgent: sampling-based continuous-action DQN
├── logging_utils.py  # pluggable metric logging (TensorBoard / null)
├── trainer.py        # self-play loop + evaluation
├── render.py         # Pygame rendering: interactive, fullscreen, or headless
└── cli.py            # `chaser-runner` entry point
configs/default.yaml  # all hyperparameters live here
tests/                # pytest suite for env, replay, agent, config
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install          # auto-run ruff + formatting on commit
pytest                      # run the test suite
ruff check . && ruff format --check .
```

CI (GitHub Actions) runs the linter, format check, and tests on every push and PR.

## Docker (headless)

The image runs training, evaluation, and GIF recording without a display. (The
live `play` window needs a display, so run that natively.)

```bash
docker build -t chaser-runner .
docker run --rm chaser-runner eval --episodes 20
docker run --rm -v "$PWD/assets:/app/assets" chaser-runner record --out assets/demo.gif
```

## Roadmap ideas

- Egocentric per-agent state (each agent sees the world from its own frame)
- Prioritized experience replay; DDPG/TD3-style deterministic actors
- Multiple chasers or runners (team pursuit-evasion)
- Optional MLflow backend behind the existing `MetricLogger` interface

## License

[MIT](LICENSE)
