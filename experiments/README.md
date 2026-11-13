# Experiment index

`before_pruning/` and `with_pruning/` contain the exact cached-score results for the unpruned baseline and the submitted classifier. Each folder has `README.md`, `metrics.json` and a per-fact `predictions.csv`.

| Setting | Correct | Accuracy | Decision threshold | Fact pruning | Evidence pairs before early exit | Pairs with early exit, batch 16 |
|---|---:|---:|---:|---|---:|---:|
| Before pruning | 188/221 | 85.07% | 0.436032310128212 | Disabled | 10,064 | 6,469 |
| Submitted pruning defaults | 193/221 | 87.33% | 0.2535 | Recall 0.30 | 7,113 | 3,690 |
| Unpruned, same decision threshold | 183/221 | 82.81% | 0.2535 | Disabled | 10,064 | 6,237 |

The first two rows compare separately selected configurations. The last two isolate pruning at the same decision threshold. All rows use context evidence, the same data and the same NLI probability cache. Evidence-level pruning is disabled. The current fact filter prunes 72 facts, including six gold-supported facts.

Against the separately tuned baseline, the current configuration improves accuracy by 2.26 percentage points. It retains 29.32% fewer evidence pairs before early exit and uses 42.96% fewer pairs after early exit at batch size 16. These are pair-count reductions, not measured runtime speedups. The same-threshold control prevents attributing the entire tuned-config difference to pruning alone.

The current Windows outputs are under `results/current_run/part1`, `results/current_run/part2` and `results/current_run/part3`; logs are under `results/current_run/logs`. The comparison snapshots in this folder predate the Windows run and were checked against its new cache. The comparison helper uses `results/current_run/part2/unit_scores.jsonl`. Run `python experiments/reproduce_cached_comparison.py` from the project root to regenerate the comparison snapshots without downloading DeBERTa.

`results/current_run/part2/pruning_comparison.csv` contains the family-selected alternatives A-D. Its B row uses IDF recall 0.25 and reaches 194/221. That row is not the submitted recall-0.30 configuration. The current defaults are described in the original summary's “Are the current thresholds still optimal?” section.

Actual Windows timing and memory measurements are in `results/current_run/part2/timing.csv`. Configuration B uses IDF recall 0.25; its timings must not be labelled as measurements of the submitted recall-0.30 defaults.

The source verification records establish unchanged executable syntax in the two cleaned Python files. The supplied driver and requirements are unchanged. Test and reproduction logs document the checks performed for this package; see `VERIFICATION.md` for scope and limitations.

## Optional parsing hybrid

`parsing/evaluate_parsing.py` evaluates the hybrid, and `parsing/test_parsing.py` checks its behavior. New results are under `results/current_run/parsing/`: 185/221 (83.71%) development accuracy and 80.00% grouped validation selecting weight and threshold inside training folds. The original dependency-only 151/221 results remain under `results/current_run/parsing_baseline/`. This is a dependency-and-lexical hybrid, not an 83.71% pure parser. The six nonzero dependency-weight candidates are 0.1, 0.2, 0.3, 0.4, 0.5 and 0.75; endpoints 0 and 1 are comparators. Existing graded outputs are unchanged. See `parsing/README.md` for full details.

## Split runner

Classifier definitions now live in standalone `factcheck.py`; experiment and self-test functions live in `factcheck_experiments.py`. Both cached-comparison and parsing-evaluation scripts import private analysis helpers from the runner. All previous result files remain unchanged. The runner defaults to the included current-run paths. See SETUP.md for the updated commands.
