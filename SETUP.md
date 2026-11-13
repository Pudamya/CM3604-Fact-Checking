# VS Code setup and commands

Open the project folder containing `factcheck.py`, `factcheck_experiments.py`, `factchecking_main.py` and `requirements.txt`.

## Python 3.10 environment

Keep your existing `.venv` when applying the update. For a fresh folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Select `.venv\Scripts\python.exe` in VS Code using Python: Select Interpreter. Install matplotlib separately if you need regenerated figures.

## Graded classifiers: unchanged

```powershell
.\.venv\Scripts\python.exe factchecking_main.py --mode word_overlap
.\.venv\Scripts\python.exe factchecking_main.py --mode entailment
```

Expected supplied-data results are 184/221 and 193/221. The entailment command needs the pretrained checkpoint locally or internet access to download it. CUDA requires a CUDA-enabled PyTorch installation and `--cuda`; it is not needed for CPU runs.

## Experiment commands

```powershell
.\.venv\Scripts\python.exe factcheck_experiments.py selftest
.\.venv\Scripts\python.exe factcheck_experiments.py part1
.\.venv\Scripts\python.exe factcheck_experiments.py experiments
.\.venv\Scripts\python.exe factcheck_experiments.py errors
```

Part 1/2/3 outputs default to `results/current_run/part1`, `part2` and `part3`; figures go to `report/current_run_figures/`. Part 2 reuses the matching saved cache. Running these commands can overwrite existing outputs; back up manually completed error worksheets first. They do not need regeneration solely because the optional parsing mode changed.

For before/after pruning comparison use `python factcheck_experiments.py compare`; add `--out_dir results/current_run/comparison` to keep the comparison separate. Existing timing data remains in `results/current_run/part2/timing.csv`; use `python factcheck_experiments.py experiments --timing` only if you intentionally want to remeasure full-model runtime and memory.

## Optional parsing hybrid

```powershell
.\.venv\Scripts\python.exe factchecking_main.py --mode parsing
.\.venv\Scripts\python.exe experiments/parsing/evaluate_parsing.py
.\.venv\Scripts\python.exe -m unittest discover -s experiments/parsing -v
```

With en_core_web_sm 3.8.0, the new hybrid gives 185/221 (83.71%), matrix [86 26; 10 99]. Its threshold is 0.5276785714285713 and dependency weight is 0.1. The evaluation writes new parsing-only tables, predictions and selection statistics under `results/current_run/parsing/`. The previous 151/221 pure-dependency baseline is preserved under `results/current_run/parsing_baseline/`.

The hybrid combines dependency and lexical recall with negation/argument guards; describe it as a hybrid. Its development accuracy is optimistic. Grouped validation selecting both weight and threshold inside training folds gives 80.00% mean accuracy (SD 1.59%). Parser-model changes can affect results. Full method details are in `experiments/parsing/README.md`.

## Other checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/m1_preprocessing -v
.\.venv\Scripts\python.exe experiments/reproduce_cached_comparison.py
```

The comparison helper uses the saved DeBERTa cache; it does not evaluate the optional parsing hybrid. Do not change the graded classifier settings to use the optional mode.

## Submission

Submit `factcheck.py` and your completed written PDF separately as the specification requests. Exclude `.venv` and bytecode from any archive. Manual Part 3 annotations and the written PDF are still required; this optional improvement does not complete those academic deliverables.
