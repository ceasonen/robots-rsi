# Continuous-control protocol correction and exploratory results

## Why this correction was necessary

The previous branch implementation incorrectly captured a *different* environment
inside the parent policy. Its phase remained zero while evaluation progressed.
Separately, source and target labels were not used by environment reset, so
both represented the same perturbation distribution. Old repair/harm comparisons
must not be used as claims of learning improvements.

This patch (1) conditions parent on the observed phase, (2) sets phase-transition
impulse to zero in source and samples the physical impulse in target, and
(3) adds regression tests for both issues. In the subsequent PD(4,3) exploratory
setting, parent gains were selected from a small independent seed diagnostic;
this is **not** a pre-registered confirmatory result.

## CPU exploratory matrix: 5 seeds per task, 60 episodes per panel per seed

Each student receives **200 teacher transitions**. Pilot feedback-selected
methods additionally train pilot students and make validation queries, which
are not equated to fixed teachers' total computational or task-query budgets.

| Task / target successes out of 300 | Parent PD(4,3) | Fixed aggressive BC | Fixed damped BC | Feedback-selected BC |
|---|---:|---:|---:|---:|
| Impulsive | 300 | 121 | 103 | 98 |
| Sluggish | 111 | 31 | 10 | 18 |
| Precision | 300 | 71 | 51 | 89 |

| Task / source successes out of 300 | Parent PD(4,3) | Fixed aggressive BC | Fixed damped BC | Feedback-selected BC |
|---|---:|---:|---:|---:|
| Impulsive | 300 | 169 | 100 | 137 |
| Sluggish | 54 | 6 | 0 | 30 |
| Precision | 300 | 153 | 50 | 154 |

The feedback selector does **not** beat the best fixed teacher in any of
these three task types on target success. Old-capability harms are substantial:
on Impulsive it loses 163 of 300 source successes; on Precision it loses 146
of 300. This is a **negative result** for the current global-ridge-BC
student and does not demonstrate self-improvement.

The immediate research implication is to halt positive claims from this toy
system, then test local repair methods with a frozen parent and state-dependent
intervention before moving to MuJoCo/LIBERO-Pro. Do not over-interpret these
small synthetic task families as independent robot tasks.

Reproduce locally with:
```bash
cd research/continuous_control
python -m unittest discover -s tests -v
python -m teachlab.run --seeds 103,107,109,113,127 --budget 200 --eval 60 --out results/pd43_exploratory.json
```
