# CPU-first synthetic teaching research harness

> **This is NOT a MuJoCo, LIBERO-Pro, DMPEL, VLA or robot experiment.** The entire `research/cpu_first/` directory is a standalone one-dimensional two-stage control abstraction. The original audited historical `code/` and `evidence/` directories remain unchanged.

The public archive does not contain the pretrained model checkpoint, private teacher trajectories and associated assets needed to rerun the historical v40 experiments. This package uses analytic online data-generating controllers and trains linear student policies, with no fake checkpoints or stored teacher traces.

## Run on CPU

```bash
cd research/cpu_first
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m cpu_lab.run --tasks forward,reverse,tight --seeds 7,17,27 --episodes 100 --out results/development.json
```

Only NumPy is needed. The synthetic environment has three task configurations, disjoint train/validation/source-test/target-test seeds, fixed teachers (`fast`, `cautious`), a random mixture, feedback-selected future teacher data collection, and a diagnostic inverted-feedback control. The teacher styles are different **control gains** and are **not** physical homing, privileged recovery, history reset, or new skills.

## Read the metrics correctly

- The main endpoint is autonomous complete two-stage student success on fresh seeds, compared with a parent; source retention and target generalization are separated.
- Paired repair/harm counts satisfy `student_success - parent_success = repair - harm` on each panel.
- All arms receive the same **number of teacher episodes**, but may receive different numbers of action labels and optimization samples; feedback selection adds two pilot student fits and two validation panels per teacher style. Do **not** claim equal total computational budget.
- Feedback-selected performance matching the best fixed style does **not** establish learned self-improvement.
- The inverted-feedback comparison is a stress test, not a calibrated permutation-significance test.
- Results only test synthetic code/protocol. They do not establish robot control efficacy, transfer to LIBERO or publication-level novelty.

## Next scientific gate

To support an RSS/ICML claim, move to a continuous-control robot benchmark with actual held-out teacher interventions, and establish a case where *equally successful teacher interventions differ in independent student acquisition*. Compare against strong fixed demonstration generation, ordinary behavior cloning and data selection under matched costs. Historical v40 measurements stay untouched.
