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


## Additional frozen-parent gate diagnostic (exploratory)

Run `python -m teachlab.gated_probe` from `research/continuous_control`.
This uses different experiment seeds (503, 509, 521, 523, 541),
60 source and 60 target trials per seed per task, and 200 teacher
transitions per trained BC student. The parent remains unchanged by the
gate, and all phase decisions use observable features (not the shock value).
The gate is defined as phase=1 and velocity norm > threshold (0.35 or 0.60).
No feedback learning or automatic gate optimization takes place.

| Task / target success out of 300 | Parent | Global aggressive BC | Aggressive gate 0.60 | Damped gate 0.60 |
|---|---:|---:|---:|---:|
| Impulsive | 300 | 133 | 209 | 295 |
| Sluggish | 108 | 30 | 51 | 56 |
| Precision | 300 | 107 | 165 | 253 |

| Task / source success out of 300 | Parent | Global aggressive BC | Aggressive gate 0.60 | Damped gate 0.60 |
|---|---:|---:|---:|---:|
| Impulsive | 300 | 233 | 291 | 300 |
| Sluggish | 42 | 74 | 32 | 0 |
| Precision | 300 | 226 | 241 | 272 |

**Negative outcome:** Restricting intervention can reduce destructive
global replacement, but NONE of the tested gates consistently exceeds
the parent in target success. The sluggish target parent has 108/300
versus 51/300 for the aggressive gate 0.60; the impulsive target parent
has 300/300 versus 209/300 for that gate. This is not a demonstrated
repair algorithm, nor a learned gate. Gate thresholds are preliminary
ablation choices. The task families are small synthetic abstractions.

The next evidence gate requires failure-conditioned **local** correction
data and a parent that demonstrably succeeds on a nonzero fraction of
source conditions, before any new self-improvement claim.
