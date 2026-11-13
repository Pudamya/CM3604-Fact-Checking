# Optional dependency-and-lexical hybrid

The optional `--mode parsing` classifier is implemented by `DependencyRecallThresholdFactChecker` in `factcheck.py`. The class name is retained for compatibility with the unchanged supplied driver. This updated mode is explicitly a hybrid, not a pure dependency parser.

## Method

1. Clean passages with the existing evidence-cleaning function, then parse facts and evidence sentences with spaCy en_core_web_sm.
2. Extract directed `(head lemma, dependency label, dependent lemma)` triples. Normalize title-name tokens and singular personal pronouns to a biography entity marker. This is a heuristic, not full coreference resolution.
3. Exclude candidate evidence sentences with incompatible explicit predicate negation or directly reversed subject/object arguments. Negation matching uses lemma and surface-stem keys to reduce lemmatizer ambiguity.
4. The dependency component is the maximum eligible sentence-level recall of fact triples. For a relation-free fact, use content-lemma recall as its fallback.
5. The lexical component uses the existing Part 1 scorer on each passage's pooled eligible sentences, with the existing Porter preprocessing and title-token removal. The original lexical class and its settings are not modified.
6. Compute **score = 0.10 x dependency recall + 0.90 x lexical recall**. Predict S only for a positive score at or above **0.5276785714285713**; otherwise predict NS.

Parsed features and lexical preprocessing are cached. No gold labels are accessed during prediction. Weight and threshold selection are performed by the separate evaluation script.

## Actual results

| Configuration | Correct | Development accuracy |
|---|---:|---:|
| Original dependency-only baseline | 151/221 | 68.33% |
| Updated optional hybrid | 185/221 | 83.71% |
| Unchanged graded word overlap | 184/221 | 83.26% |
| Unchanged graded entailment | 193/221 | 87.33% |

The hybrid confusion matrix [TP FN; FP TN] is **[86 26; 10 99]**. It improves by 34 correct predictions over the original dependency-only baseline. Its strong lexical contribution must be disclosed; this is not evidence that pure dependency parsing reaches 83.71%.

## Selection and validation

The fixed candidate dependency weights are 0.1, 0.2, 0.3, 0.4, 0.5 and 0.75. Weight 0 (guarded lexical only) and weight 1 (guarded dependency only) are endpoint comparators, not eligible hybrid choices. All weights use the same eligible evidence and component scores.

For the deployed configuration, choose the highest fixed-configuration grouped-CV mean accuracy, using full-development accuracy and then the smaller dependency weight as tie-breaks. Select its final threshold on the full development set by maximum accuracy and widest score-gap tie-breaking. That procedure selects weight 0.1 and threshold 0.5276785714285713.

Two validation views are reported, each using 10 repeats of 5 folds grouped by biography:

- Fixed weight, with threshold retuned on training biographies: **80.00% mean accuracy, SD 1.59%**.
- Weight and threshold both chosen on training biographies before evaluating held-out biographies: **80.00% mean accuracy, SD 1.59%**.

The second view evaluates the training-fold weight/threshold selection procedure separately from the final full-data fit. The full-development 83.71% is optimistic. The validation uses the supplied development data, not an independent unseen test set; no future 80% guarantee is implied.

## Reproduce in VS Code / PowerShell

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe factchecking_main.py --mode parsing
.\.venv\Scripts\python.exe experiments/parsing/evaluate_parsing.py
.\.venv\Scripts\python.exe -m unittest discover -s experiments/parsing -v
```

New results are under `results/current_run/parsing/`: `summary.md`, `summary.json`, `predictions.csv`, `weight_comparison.csv`, `threshold_sweep.csv` and verification logs. Predictions include both component scores. The previous pure-dependency results and Windows metadata are preserved unchanged under `results/current_run/parsing_baseline/`.

The new checks used Linux/Python 3.12, spaCy 3.8.2 and en_core_web_sm 3.8.0. Other parser versions may produce different scores. All Python files also pass Python 3.10 syntax parsing; this is not a new Windows/CUDA validation.

## Tests and limitations

The 17 parsing tests cover positive and negative support, reversed roles at the deployed hybrid threshold, compatible evidence across sentences, contradictory distractors, title/pronoun substitution, empty and unrelated inputs, lexical fallback, cache stability, lemma/role extraction, invalid thresholds/weights and missing-model guidance. The 25 existing self-tests and 36 preprocessing tests also pass (78 total).

The lexical component may over-credit shared words across sentences. Exact dependency matching misses paraphrases and active/passive alternations. Pronoun substitution and negation/argument guards are shallow approximations and do not establish full semantic entailment. The hybrid still makes 36 development-set errors and does not replace DeBERTa. Graded classes, defaults, Part 1/2/3 results and the trained-score cache are unchanged.
