# Continuous-state control experiment (CPU)

This is a deterministic 2-D inertial point-mass control benchmark with two sequential goals and a hidden velocity shock at the phase transition. It is a continuous-state/action dynamical system, **not a rigid-body robot simulator, MuJoCo, LIBERO-Pro or real-robot evidence**.

## Run
```bash
python -m pip install -r research/continuous_control/requirements.txt
PYTHONPATH=research/continuous_control python -m unittest discover -s research/continuous_control/tests -v
PYTHONPATH=research/continuous_control python -m teachlab.run --seeds 3,7,11,17,23 --budget 200 --eval 30
```

## Experimental protocol
- State observations: 2-D position, velocity, phase, normalized time and bias. The phase shock is hidden.
- Teachers: aggressive PD and damped PD; students use the same fixed nonlinear feature map and ridge regression.
- Each training arm receives exactly B teacher transitions. Feedback arms allocate 20% to a balanced pilot and 80% to the selected teacher; fixed arms use B transitions of one teacher. Final evaluation uses disjoint source and target seed panels.
- Metrics report complete two-stage success and paired repair/harm separately.
- The feedback inverted arm is a falsification stress test, not a valid permutation test.

## Initial CPU result
Five seeds (3, 7, 11, 17, 23), B=200 transitions per arm, 30 source and 30 target evaluation episodes per seed. Mean target success across seeds:
| Task | Fixed aggressive | Fixed damped | Feedback selected | Inverted feedback |
|---|---:|---:|---:|---:|
| impulsive | 0.420 | 0.327 | 0.360 | 0.120 |
| sluggish | 0.067 | 0.073 | 0.047 | 0.047 |
| precision | 0.207 | 0.180 | 0.267 | 0.087 |

The validation selector chose aggressive in all 15 task-seed runs. Thus this experiment **does not demonstrate adaptive teacher selection**. Feedback-selected was lower than the best fixed teacher on impulsive and sluggish, and higher on precision; the small evaluation panels and low success rates make these exploratory only. The inverted control is worse, but that alone does not establish a valid causal feedback effect.

## Limitations / next gate
The continuous dynamics are hand-designed and lack contacts, object geometry, camera observations, manipulation constraints and a validated robot task distribution. The current student is a linear ridge model, not a visuomotor policy. MuJoCo/Gymnasium could not be installed in this execution environment because package network access was unavailable. Do not claim RSS-level evidence from these results.

Next gate: a physics-based manipulation benchmark with real contact-rich teacher interventions, matched *total* interaction and compute budgets, task-held-out validation, multiple independent training seeds, and a preregistered statistical comparison. Historical v40 assets and evidence remain untouched.


## Protocol correction: phase and domain split

The earliest continuous-control runs had two significant protocol errors:
the parent controller referenced a separate environment that never advanced
to phase two, and the evaluation accepted “source”/“target” labels without
changing the transition dynamics. Those earlier numbers are **invalid as
repair/retention evidence**. This update makes the parent phase-conditioned
on its actual observation and uses **zero shock for source**, **sampled shock
for target**, while retaining identical initial positions for paired seeds.

As a development-only positive-retention check, the parent PD was set to
(4.0, 3.0), chosen after an independent exploratory seed sweep; this is not
a frozen benchmark. Both the weak-parent and PD(4,3) runs and their caveats
are summarized in [the protocol audit](PROTOCOL_AUDIT.md).

Run the corrected tests with:

```bash
cd research/continuous_control
python -m unittest discover -s tests -v
python -m teachlab.run --seeds 103,107,109,113,127 --budget 200 --eval 60 --out results/pd43_exploratory.json
```

No result from this CPU point-mass abstraction establishes robot RSI, RSS
readiness, or an emergent autonomous learner.
