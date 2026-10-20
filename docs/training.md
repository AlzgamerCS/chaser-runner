# Training diagnosis and validation

The original code combined several sources of difficulty:

- The policy and Bellman maximum both searched fresh random continuous actions.
  With two independently changing opponents, the Q estimates and trajectories
  could be noisy even with exploration disabled.
- Actions changed at 60 Hz, while inertia makes the effect of one acceleration
  small. A discount of 0.99 at this rate has an effective horizon of roughly
  100 ticks (1.7 seconds), shorter than many useful pursuit maneuvers.
- Distance reward was divided by an 800-unit world size. Typical per-tick rewards
  were small compared with function approximation error. Reward scaling is now
  configurable, and the default is larger; old and new returns are not directly
  comparable.
- Both policies updated throughout every encounter. Training now freezes the
  opponent for an entire episode and mixes historical and reference opponents.
  This reduces one source of nonstationarity; it does not make all self-play
  stationary or guarantee convergence.
- Timeout and capture shared a terminal mask. The new objective treats the time
  limit as an external rollout cutoff and bootstraps through it. This is an
  intentional change from treating timeout as an absorbing runner win. If a
  future objective adds an actual finite-horizon win reward, remaining time must
  be represented in the observation and its terminal handling revisited.
- Evaluation seeded initial positions but not sampled actions, printed only a
  paired score, and discarded the results. Fixed independent reference opponents
  now make it possible to see which role improves.
- Weight files overwrote one another and did not preserve optimizers, targets,
  replay, counters, episode state, or RNGs. Weight-only continuation was a warm
  start rather than a continuation of the experiment.

The replacement continuous learner follows TD3: twin critics with a minimum
bootstrap target, target policy smoothing, delayed actor updates, and soft target
updates. The original DQN remains available with Double DQN selection/evaluation
and Huber loss. These changes are based on the
[TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a.html) and the
[Gymnasium explanation of time-limit bootstrapping](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

The action hold is four physics ticks by default. At 15 policy decisions per
second, gamma=0.99 gives a horizon of roughly 6.7 seconds. This is an approximation
using 1/(1-gamma), not a guarantee of long-term planning. The distance reward is a
shaped training objective rather than a policy-invariant potential-based shaping
guarantee. Increasing return alone does not establish better capture/evasion.

The tests establish deterministic resume on the same CPU runtime, isolated and
repeatable evaluation, correct terminal masks, artifact preservation, and actual
learning on a known continuous-action objective. They do not establish that the
self-play game converges. CUDA continuation preserves RNG state but has not been
verified here, and hardware or runtime changes can affect numerical behavior.

Short experiments should be assessed using both capture rate and episode length,
against several opponent types. Save the best policy before later updates lose
an earlier skill. Validation seed ranges are reused for selection, so final
reports must use held-out episode seeds and multiple independent training seeds.
Wilson intervals quantify episode sampling uncertainty for a policy; they do not
measure variability between independently trained policies.

## Observed CPU experiments (2026-10-20)

Two independent training seeds ran for 50,000 policy decisions each using the
new defaults, with 20 validation episodes per matchup every 10,000 decisions.
Each role's best policy was selected using validation results, before testing on
100 new episodes per matchup starting at seed 20,000. Both runs selected chaser
step 30,000 and runner step 20,000. Values below compare the untrained network
with these selected policies, not the latest training step.

| Training seed | Chaser captures scripted evader | Chaser captures random runner | Runner survives scripted pursuit | Runner duration against scripted pursuit |
| --- | --- | --- | --- | --- |
| 0 | 0% → 100% | 15% → 85% | 0% → 48% | 86 → 695 physics ticks |
| 1 | 0% → 96% | 12% → 86% | 0% → 4% | 86 → 376 physics ticks |

Pursuit improved on both training seeds. Evasion improved unevenly. For context,
the scripted runner against the scripted chaser survived 0% of these episodes
and lasted 404 ticks on average. The seed-0 selected runner exceeded that baseline;
the seed-1 runner did not on mean duration. At step 50,000, both learned runners
had fallen back to 0% survival, while chaser capture against the scripted evader
was 83% and 74%. This demonstrates why latest weights and best validation policies
must be preserved separately. It also shows that stable self-play convergence
and seed-independent evasion remain unresolved.

[Raw results](learning-validation.json) include the full configuration, selected
steps, all matchups for untrained/latest/selected policies, and episode confidence
intervals. The run directories and their policy/checkpoint artifacts are local,
Git-ignored outputs. Reproduce the held-out comparison with:

```bash
python scripts/benchmark_runs.py runs/validated-seed0 runs/validated-seed1 \
    --episodes 100 --seed 20000 --out evaluations/learning-report.json
```

These are short development experiments on Python 3.14.6, PyTorch 2.14.1+cpu,
NumPy 2.5.4, Windows, using one PyTorch thread per process. They are not an
ablation study against the previous implementation: algorithm, reward scale,
action timing, and opponent schedule changed together. More training seeds and
additional fixed historical opponents are needed to judge broader generalization.
