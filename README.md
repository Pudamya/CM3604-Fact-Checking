# CM3604 Fact Checking

The implementation is split into a standalone classifier file and a separate experiment runner. Existing classifier behavior, thresholds, data and recorded results are preserved.

## Project layout

| Path | Purpose |
|---|---|
| `factcheck.py` | Standalone classifiers: preprocessing, lexical metrics, word overlap, DeBERTa entailment and optional parsing |
| `factcheck_experiments.py` | Part 1/2 experiments, timing, comparison, error worksheets and self-tests; imports the classifiers |
| `factchecking_main.py` | Supplied graded driver, unchanged byte for byte |
| `requirements.txt` | Supplied pinned dependencies, unchanged |
| `data/` | Supplied facts, retrieved passages and dataset licence |
| `experiments/m1_preprocessing/` | Preprocessing and lexical-metric tests |
| `experiments/parsing/` | Optional parsing evaluation, tests and method documentation |
| `experiments/reproduce_cached_comparison.py` | Reproduce existing pruning comparisons using saved scores |
| `experiments/before_pruning/`, `with_pruning/`, `unpruned_same_threshold/` | Earlier comparison snapshots |
| `results/current_run/part1/` | Metric, preprocessing and threshold comparisons |
| `results/current_run/part2/` | Pruning experiments, timing measurements and DeBERTa score cache |
| `results/current_run/part3/` | Error worksheets requiring manual category and notes entries |
| `results/current_run/parsing/` | New hybrid results, weight comparison and regression logs |
| `results/current_run/parsing_baseline/` | Preserved dependency-only baseline results and logs |
| `results/current_run/logs/` | Recorded Windows experiment logs |
| `report/current_run_figures/` | Current-run report figures |
| `report/figures/` | Earlier comparison figures retained for reference |
| `experiments/verification/` | Historical verification records plus the new split verification |

`factcheck.py` does not import `factcheck_experiments.py` or any supporting project script. The graded commands require only the submitted classifier file, supplied driver, data and installed dependencies. The optional parser also requires the spaCy English model. Experiment scripts use the separate runner and do not need to be submitted as the classifier implementation.

## Results and current settings

| Classifier/configuration | Correct | Development accuracy |
|---|---:|---:|
| Word overlap | 184/221 | 83.26% |
| Entailment: separately tuned unpruned baseline | 188/221 | 85.07% |
| Entailment: current recall-pruned defaults | 193/221 | 87.33% |
| Optional parsing hybrid | 185/221 | 83.71% |
| Original dependency-only parsing baseline | 151/221 | 68.33% |

Current entailment settings are decision threshold 0.2535, fact-level recall pruning 0.30, evidence pruning disabled, context evidence and early exit enabled. The optional hybrid uses 10% dependency recall plus 90% lexical recall, with threshold 0.5276785714285713. The unpruned entailment baseline uses decision threshold 0.436032310128212. An unpruned control at threshold 0.2535 achieves 183/221 and isolates pruning at the same decision threshold.

The IDF-recall alternative in Part 2 reaches 194/221 with fact pruning 0.25. It is an experiment and has not been applied to the submitted classifier. Its timing rows must not be presented as timings of the current recall-0.30 defaults.

These are development results, not unseen-test results. Grouped cross-validation is labelled separately. The optional hybrid improves over the dependency-only baseline on this development set but remains below DeBERTa. It is not presented as pure dependency parsing. Earlier recorded results, logs and figures are retained unchanged after this refactor.

## Run

Follow [SETUP.md](SETUP.md) for Python 3.10 environment setup and commands.

```powershell
.\.venv\Scripts\python.exe factchecking_main.py --mode word_overlap
.\.venv\Scripts\python.exe factchecking_main.py --mode entailment
.\.venv\Scripts\python.exe factchecking_main.py --mode parsing
```

Experiments now run through the separate file:

```powershell
.\.venv\Scripts\python.exe factcheck_experiments.py selftest
.\.venv\Scripts\python.exe factcheck_experiments.py part1
.\.venv\Scripts\python.exe factcheck_experiments.py experiments
.\.venv\Scripts\python.exe factcheck_experiments.py errors
```

The experiment runner defaults to the included `results/current_run/` directories and `report/current_run_figures/`. The DeBERTa scoring cache remains valid because the evidence and inference method source has not changed.

## Pruning locations

In `factcheck.py`, find `PART 2: ENTAILMENT CLASSIFIER AND PRUNING`, then `PRUNING SETTINGS`. `_prepare()` is marked `FACT-LEVEL PRUNING`; `plan_evidence()` is marked `EVIDENCE-LEVEL PRUNING`. Changing the optional parsing classifier does not change entailment or word overlap.

## Updated optional parsing mode

`--mode parsing` now selects the dependency-and-lexical hybrid. The class name stays `DependencyRecallThresholdFactChecker` for compatibility with the unchanged supplied driver. The hybrid uses the existing lexical scorer without modifying it, and filters candidate sentences with explicit incompatible negation or directly reversed subject/object roles before scoring.

Actual new result: **185/221 (83.71%)**, confusion matrix [86 26; 10 99]. Grouped cross-validation selecting both dependency weight and threshold inside the training biographies gives **80.00% mean accuracy (SD 1.59%)**, across 10 repeats of 5 folds. The final full-development score is optimistic and not an unseen-test guarantee. The candidate comparison and method limitations are recorded in `results/current_run/parsing/` and `experiments/parsing/README.md`.

The original 151/221 dependency-only results are preserved in `results/current_run/parsing_baseline/`. The graded word-overlap and entailment classifiers, their thresholds, pruning settings and Part 1/2/3 outputs are unchanged. Their previously measured results remain 184/221 and 193/221 respectively.
