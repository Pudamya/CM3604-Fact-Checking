# Verification and run provenance

The package contains a current Windows run and historical Linux verification. Keep these evidence sources separate when reporting results.

## Current Windows run: 13 November 2026

The supplied Windows environment identifies Python **3.10.11**. Installed package metadata matches the pinned requirements: NumPy 2.0.2, torch 2.5.0, torchvision 0.20.0, tqdm 4.66.5, spaCy 3.8.2, NLTK 3.9.1, transformers 4.45.2, sentencepiece 0.2.0, protobuf 5.28.3 and rouge-score 0.1.2. This metadata was inspected in the uploaded archive; a fresh Windows environment was not independently recreated during this documentation update.

| Check or measurement | Evidence and result |
|---|---|
| Part 1 classifier | `results/current_run/part1/summary.md`: 184/221; confusion matrix [86 26; 11 98] |
| Graded entailment command | Supplied terminal transcript: 193/221; confusion matrix [94 18; 10 99] |
| Part 2 experiment run | `results/current_run/logs/part2.log`: scoring completed for 221 facts, then all four pruning families evaluated |
| Cache | `results/current_run/part2/unit_scores.jsonl`: 221 records, 112 gold S and 109 gold NS, fingerprint `9c4214be021fb3b8` |
| Unpruned baseline | Current cache replay: 188/221 at threshold 0.436032310128212 |
| Same-threshold unpruned control | Current cache replay: 183/221 at threshold 0.2535 |
| Submitted recall-pruned defaults | Earlier prediction snapshot checked against the current cache: 193/221 |
| Timing and memory | `results/current_run/part2/timing.csv`: 11 measurements; every row reports `matches_offline_analysis=True` |
| Part 3 | `results/current_run/logs/part3.log`: 193/221, 10 FP and 18 FN; both worksheets generated |
| Current figures | `report/current_run_figures/`: three Part 1 and three Part 2 figures |

The supplied logs document CPU runs. CUDA was not tested. Full trained-model scoring and timing were performed in the Windows run; these are no longer merely unmeasured cached-score estimates.

## Configuration-specific timing

| Configuration | Accuracy | Wall time (s) | Peak process RSS (MB) |
|---|---:|---:|---:|
| A: no pruning, batch 16 | 85.07% | 1180.5 | 1577 |
| B: IDF recall fact pruning 0.25 | 87.78% | 795.0 | 1737 |
| C: evidence pruning | 86.88% | 300.4 | 1483 |
| D: fact and evidence pruning | 86.88% | 316.0 | 1612 |
| B with early exit | 87.78% | 592.9 | 1735 |

Times and RSS are rounded from the supplied Windows output. Peak RSS includes model weights. The first five timing rows compare batch sizes on 30 facts; full-configuration rows use 221 facts. B uses IDF pruning, whereas the submitted classifier uses recall pruning 0.30 and reaches 193/221. Do not attribute B's accuracy or timings to the submitted defaults. No full timing row specifically measures the submitted recall-0.30 configuration.

## Historical Linux verification

`experiments/verification/environment.json` describes the earlier Linux environment: Python 3.12.14, CPU torch 2.5.0+cpu and NumPy 2.3.5. It is retained as historical evidence and does not describe the later Windows environment.

| Earlier check | Recorded result |
|---|---|
| Self-tests | 25 passed, including tiny-model driver and command smoke tests |
| Preprocessing tests | 36 passed after installing the spaCy English model |
| Word-overlap driver | 184/221 |
| Cached comparisons | 188/221 baseline, 193/221 submitted defaults, 183/221 same-threshold unpruned control |
| Source preservation | Executable AST unchanged after ignoring docstrings; cache-sensitive inference/evidence source preserved |
| Python 3.10 syntax | All project Python files parsed using Python 3.10 syntax rules |

These historical test-suite logs are not evidence of a new Windows execution of all 61 tests. They remain unchanged. The Windows full-model runs and measurements are documented separately above.

## File provenance and remaining work

- `factcheck.py`, `factchecking_main.py` and `requirements.txt` retain the previously checked executable code, driver and package pins.
- `data/` contains the supplied facts, passages and dataset licence.
- `results/current_run/part1/`, `part2/` and `part3/` contain the supplied current Windows outputs; `results/current_run/logs/` contains their terminal logs.
- `experiments/before_pruning/`, `with_pruning/` and `unpruned_same_threshold/` are earlier cache-derived comparison snapshots. Their predictions were checked against the current cache; their probability values and provenance should not be relabelled as newly measured Windows outputs.
- `report/current_run_figures/` contains Windows experiment figures. `report/figures/` contains earlier comparison figures.
- `experiments/verification/file_sha256.json` is a historical inventory, not a current integrity manifest after documentation changes. `source_verification.json` describes the earlier source cleanup. Neither was rewritten to imply a new test run.
- The saved-score comparison helper uses `results/current_run/part2/unit_scores.jsonl`. The optional parsing class has now been implemented; the graded classifiers, cache-sensitive methods and supplied driver remain unchanged.
- The 20 selected Part 3 errors still need manual category and notes entries. A completed written-submission PDF is not included.

## Historical dependency-only implementation and checks

The optional class now extracts labelled dependency triples and scores maximum sentence-level dependency recall. It includes lemma normalization, page-title/pronoun entity substitution, explicit predicate-polarity checks, a content-lemma fallback for relation-free facts, input validation and a parsed-text cache. It requires the spaCy English parser model. Full details and limitations are in `experiments/parsing/README.md`.

New checks used Linux, Python 3.12, NumPy 2.0.2, CPU torch 2.5.0+cpu, spaCy 3.8.2 and en_core_web_sm 3.8.0. They do not establish a new Windows or CUDA validation.

| Check | Result |
|---|---|
| Existing self-tests | 25 passed |
| Existing preprocessing tests | 36 passed |
| New parsing-specific tests | 13 passed |
| Real parsing driver on all supplied facts | 151/221 (68.33%); confusion matrix [74 38; 32 77] |
| Real word-overlap driver | 184/221; confusion matrix [86 26; 11 98] |
| Existing trained entailment cache replay | 193/221; confusion matrix [94 18; 10 99] |
| Source isolation | Removing the optional parsing class yields identical module ASTs before and after the change |
| Driver preservation | `factchecking_main.py` unchanged byte for byte from the uploaded project |
| Cache fingerprint source | All seven inference/evidence methods preserved exactly |
| Syntax | All project Python files parse under Python 3.10 syntax rules |

The full pretrained DeBERTa checkpoint was not rerun during this update. Its unchanged code, the cached-score regression and existing full Windows run evidence support the retained result. Existing Part 1, Part 2, Part 3, timing, figures and historical log files were not regenerated. New evaluation artifacts are separate under `results/current_run/parsing/`.

The parsing threshold was selected on the supplied development set, where it reaches 151/221. Grouped cross-validation retuning the threshold on training biographies gives 62.76% mean accuracy across 10 repeats of 5 folds (SD 1.64%). This is an optional structural baseline with lower accuracy than the graded classifiers; it is not presented as an improvement over DeBERTa.

## Historical split-file verification

The latest organization moves experiment commands and self-tests into `factcheck_experiments.py`. All classifier definitions, including parsing, remain in standalone `factcheck.py`; their function/class ASTs and the supplied driver are unchanged from the uploaded project. Supporting cached-comparison and parsing-evaluation imports have been updated. The new runner defaults to the included `results/current_run/` folders.

New verification: 74 tests passed, standalone word-overlap and parsing driver runs matched 184/221 and 151/221, a standalone tiny-model entailment smoke test passed, and the cached DeBERTa comparison matched 193/221. The experiment commands and subprocess timing boundary were checked. Full details, scope and logs are in `experiments/verification/split/README.md` and `summary.json`. Older verification records remain historical; the full pretrained model and timing benchmark were not rerun for this reorganization.

## Current optional hybrid update

Only the optional parsing class and its evaluation/tests/documentation were changed. Every other classifier function/class has an identical AST to the uploaded split project. `factchecking_main.py` remains byte-identical to the original driver; the word-overlap and entailment settings and all cache-sensitive evidence/inference method source are unchanged.

The deployed hybrid is 0.1 times dependency recall plus 0.9 times lexical recall, threshold 0.5276785714285713. Both components use eligible evidence; explicit incompatible predicate negation and reversed subject/object candidates are rejected. The lexical scorer is reused without changing its class or preprocessing. Negation comparison includes lemma and surface-stem predicate keys to reduce lemmatizer ambiguity.

New full parsing inference gives 185/221 (83.71%), matrix [86 26; 10 99], compared with the preserved original dependency-only baseline 151/221 (68.33%). Fixed-weight grouped validation and training-fold joint selection of weight and threshold each give 80.00% mean accuracy, SD 1.59%, across 10 repeats of 5 folds grouped by person. These remain development-data validation procedures, not a separate unseen test set. All candidate values and predictions are included under `results/current_run/parsing/`.

Regression checks: 17 parsing tests, 36 preprocessing tests and 25 existing self-tests pass (78 total); the word-overlap driver gives 184/221 and the unchanged trained DeBERTa cache reproduces 193/221. No new full pretrained DeBERTa inference or timing benchmark was run. Existing Part 1, Part 2, Part 3, timing and figure files are preserved byte for byte. Original parsing artifacts were copied intact into `results/current_run/parsing_baseline/` before generating the new results.

New checks used Linux/Python 3.12, spaCy 3.8.2 and en_core_web_sm 3.8.0. They do not claim a new Windows or CUDA validation. All Python files also parse under Python 3.10 syntax rules. Historical verification inventories describe earlier versions; the latest hybrid `change_verification.json` records the scope of this change.
