# Split-file verification

The uploaded monolithic source was reorganized into standalone `factcheck.py` and `factcheck_experiments.py`. Classifier function/class ASTs are identical to the upload, including the optional parser. All experiment imports and helper entry points were checked. The original driver, dependencies, data, results, figures and prior logs are preserved.

## Current checks

- 25 existing self-tests, 36 preprocessing tests and 13 parsing tests passed (74 total).
- An isolated folder containing only the classifiers, unmodified driver and data produced 184/221 in word-overlap mode and 151/221 in parsing mode.
- The same isolated folder completed entailment through the driver with a tiny random model and no experiment runner. This checks execution, not pretrained-model accuracy.
- The cached comparison helper ran in a separate temporary mirror with the new imports and reproduced 188/221 unpruned, 193/221 current defaults and 183/221 at the same threshold without pruning.
- The actual `part1`, `experiments` and `errors` commands ran through the new runner into temporary verification directories. Seven Part 2 CSVs match byte for byte; Part 3 worksheets match after normalizing Windows/Linux newlines. Part 1 non-timing metrics match to floating-point precision; two standard-deviation values differ only in their last floating-point digits.
- The existing self-tests exercised `compare` with a tiny model. A real subprocess timing-dispatch/CLI/worker check also passed with a tiny model; it is not a trained-model benchmark.
- The parsing evaluator ran with its new imports and reproduced all 221 parsing predictions exactly.
- The score-cache fingerprint remains compatible. No pretrained DeBERTa re-scoring or full timing benchmark was required or rerun.
- All project Python files pass Python 3.10 syntax parsing. Runtime checks here used Linux and Python 3.12, not a new Windows/CUDA run. See summary.json for installed versions.

The new commands default to `results/current_run/` and `report/current_run_figures/`. Preserved historical logs and generated summaries may show old command names and earlier paths; they remain historical evidence, not the current setup guide. See the root SETUP.md for current commands.

Part 3 manual category/notes entries and the written-submission PDF remain outstanding. This refactor does not complete those academic deliverables.
