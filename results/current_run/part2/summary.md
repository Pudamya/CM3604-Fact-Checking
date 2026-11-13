# Part 2 experiment results

Facts: 221 (112 S, 109 NS incl. IR), 32 people (CV groups). Cached units: {'sentence': 5844, 'context': 4220, 'titled': 1037, 'passage': 692}. Cache fingerprint `9c4214be021fb3b8`.

All numbers are on the supplied development set (the only labelled data). **cv** = cross-validation grouped by person, 10 repeats x 5 folds: the pruning setting and the decision threshold are re-tuned on the training people and the held-out people are predicted, so cv numbers are the honest estimate. **dev** = the final setting and threshold applied to all 221 facts (optimistic). **pairs** = premise/hypothesis pairs sent to DeBERTa for all 221 facts; with early exit, scoring stops as soon as one unit reaches the threshold (same predictions).

## A-D: pruning configurations

| config | final setting | threshold | cv_acc (nested) | cv_acc_sd | cv_acc (final setting fixed) | cv S_precision | cv S_recall | cv S_f1 | cv NS_f1 | cv macro_f1 | cv CM [TP FN; FP TN] per repeat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no pruning (baseline) | no pruning | 0.436 | 82.0% | 2.4% | 82.0% | 83.4% | 80.5% | 82.0% | 82.1% | 82.0% | [90 22; 18 91] |
| B: fact-level pruning only | fact idf_recall>=0.25 | 0.2535 | 83.9% | 1.4% | 87.3% | 85.7% | 81.9% | 83.7% | 84.0% | 83.9% | [92 20; 15 94] |
| C: evidence-level pruning only | evidence lexical fuzzy_recall>=0.20 | 0.408 | 82.5% | 1.5% | 85.7% | 86.3% | 77.9% | 81.9% | 83.1% | 82.5% | [87 25; 14 95] |
| D: fact-level + evidence-level pruning | fact recall>=0.20, evidence safe recall>=0.20 | 0.325 | 81.4% | 0.9% | 86.1% | 83.5% | 78.8% | 81.1% | 81.6% | 81.4% | [88 24; 18 92] |

| config | dev_acc | dev S_precision | dev S_recall | dev S_f1 | dev NS_precision | dev NS_recall | dev NS_f1 | dev macro_f1 | dev CM [TP FN; FP TN] | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no pruning (baseline) | 85.1% | 85.6% | 84.8% | 85.2% | 84.5% | 85.3% | 84.9% | 85.1% | [95 17; 16 93] | 16 | 17 |
| B: fact-level pruning only | 87.8% | 91.3% | 83.9% | 87.4% | 84.7% | 91.7% | 88.1% | 87.8% | [94 18; 9 100] | 9 | 18 |
| C: evidence-level pruning only | 86.9% | 91.1% | 82.1% | 86.4% | 83.3% | 91.7% | 87.3% | 86.9% | [92 20; 9 100] | 9 | 20 |
| D: fact-level + evidence-level pruning | 86.9% | 91.9% | 81.2% | 86.3% | 82.8% | 92.7% | 87.4% | 86.9% | [91 21; 8 101] | 8 | 21 |

| config | facts pruned | S facts pruned | pairs | pairs_saved | pairs with early exit | early_saved | model inputs (windows) | premise tokens |
|---|---|---|---|---|---|---|---|---|
| A: no pruning (baseline) | 0 | 0 | 10064 | 0.0% | 6469 | 35.7% | 10064 | 333011 |
| B: fact-level pruning only | 74 | 6 | 7026 | 30.2% | 3633 | 63.9% | 7026 | 237419 |
| C: evidence-level pruning only | 0 | 0 | 1873 | 81.4% | 1715 | 83.0% | 1873 | 81630 |
| D: fact-level + evidence-level pruning | 46 | 4 | 1924 | 80.9% | 1756 | 82.6% | 1924 | 83418 |

Previous submission (context, no pruning, t=0.436): dev 188/221 = 85.07%, cv 82.04%, 10064 pairs.

Errors of each final setting compared with A: A 33 errors (0 fixed, 0 new); B 27 errors (11 fixed, 5 new); C 29 errors (7 fixed, 3 new); D 29 errors (9 fixed, 5 new). Lists: errors_by_config.csv.

## Accuracy versus cost

Every setting searched (accuracy_vs_cost_all_settings.csv); best grouped-CV accuracy for a given budget:

| family | setting | cv_acc | pairs | pairs_saved |
|---|---|---|---|---|
| B | fact tfidf_cosine>=0.50 | 39.1% | 41 | 99.6% |
| B | fact tfidf_cosine>=0.40 | 39.5% | 134 | 98.7% |
| B | fact tfidf_cosine>=0.34 | 41.2% | 270 | 97.3% |
| B | fact tfidf_cosine>=0.30 | 45.8% | 654 | 93.5% |
| C | evidence lexical recall>=0.50 | 81.1% | 772 | 92.3% |
| D | fact recall>=0.50, evidence lexical recall>=0.34 | 81.8% | 804 | 92.0% |
| C | evidence lexical recall>=0.34 | 82.2% | 826 | 91.8% |
| C | evidence lexical fuzzy_recall>=0.50 | 82.5% | 861 | 91.4% |
| C | evidence lexical fuzzy_recall>=0.34 | 83.2% | 922 | 90.8% |
| D | fact recall>=0.34, evidence safe recall>=0.50 | 83.5% | 999 | 90.1% |
| D | fact recall>=0.34, evidence safe recall>=0.34 | 83.7% | 1028 | 89.8% |
| D | fact fuzzy_recall>=0.30, evidence lexical fuzzy_recall>=0.25 | 84.2% | 1557 | 84.5% |
| C | evidence lexical recall>=0.25 | 85.2% | 1577 | 84.3% |
| D | fact recall>=0.25, evidence lexical recall>=0.20 | 85.6% | 1692 | 83.2% |
| D | fact fuzzy_recall>=0.30, evidence lexical fuzzy_recall>=0.20 | 85.6% | 1712 | 83.0% |
| D | fact fuzzy_recall>=0.25, evidence lexical fuzzy_recall>=0.20 | 85.7% | 1844 | 81.7% |
| D | fact recall>=0.25, evidence safe recall>=0.20 | 85.9% | 1864 | 81.5% |
| D | fact recall>=0.20, evidence safe recall>=0.20 | 86.1% | 1924 | 80.9% |
| B | fact idf_recall>=0.25 | 87.3% | 7026 | 30.2% |

## Evidence selection and decision strategy (no pruning)

| evidence units | threshold | dev_acc | cv_acc | FP | FN | pairs | model inputs |
|---|---|---|---|---|---|---|---|
| sentence | 0.3308 | 83.3% | 80.0% | 15 | 22 | 5844 | 5844 |
| sentence + pronoun-resolved | 0.3308 | 83.7% | 80.9% | 15 | 21 | 6881 | 6881 |
| sentence + previous-sentence context | 0.436 | 85.1% | 82.0% | 16 | 17 | 10064 | 10064 |
| context + pronoun-resolved | 0.331 | 85.1% | 81.7% | 18 | 15 | 11101 | 11101 |
| whole passage | 0.0211 | 79.6% | 72.7% | 35 | 10 | 692 | 692 |
| context, decision = entailment_minus_contradiction | 0.434 | 85.1% | 81.4% | 16 | 17 | 10064 | 10064 |
| context, decision = argmax_any | 0.5 | 84.2% | 84.2% | 16 | 19 | 10064 | 10064 |

## Does pruning remove relevant evidence or supported facts?

Fact level: supported facts pruned. Evidence level: facts predicted S without pruning whose best evidence unit is removed (their score drops). 'safe' keeps the 3 highest-overlap units and the article's lead sentence; 'fuzzy_recall' also matches 5-letter prefixes.

| level | metric | rule | tau | units kept | facts pruned | S facts pruned | best evidence lost (S-predicted facts) | of which gold S | cv_acc |
|---|---|---|---|---|---|---|---|---|---|
| fact | recall |  | 0.05 | 82.6% | 43 | 4 |  |  | 85.2% |
| fact | recall |  | 0.1 | 82.6% | 43 | 4 |  |  | 85.2% |
| fact | recall |  | 0.2 | 80.7% | 46 | 4 |  |  | 85.2% |
| fact | recall |  | 0.25 | 77.4% | 55 | 4 |  |  | 86.2% |
| fact | recall |  | 0.3 | 70.7% | 72 | 6 |  |  | 86.8% |
| fact | recall |  | 0.34 | 66.8% | 82 | 7 |  |  | 84.9% |
| fact | recall |  | 0.4 | 66.8% | 82 | 7 |  |  | 84.9% |
| fact | recall |  | 0.5 | 63.4% | 89 | 9 |  |  | 84.5% |
| fact | fuzzy_recall |  | 0.05 | 86.8% | 34 | 3 |  |  | 84.4% |
| fact | fuzzy_recall |  | 0.1 | 86.8% | 34 | 3 |  |  | 84.4% |
| fact | fuzzy_recall |  | 0.2 | 85.1% | 36 | 3 |  |  | 84.4% |
| fact | fuzzy_recall |  | 0.25 | 82.8% | 42 | 3 |  |  | 84.5% |
| fact | fuzzy_recall |  | 0.3 | 76.4% | 58 | 3 |  |  | 85.8% |
| fact | fuzzy_recall |  | 0.34 | 72.3% | 69 | 4 |  |  | 85.3% |
| fact | fuzzy_recall |  | 0.4 | 72.3% | 69 | 4 |  |  | 85.3% |
| fact | fuzzy_recall |  | 0.5 | 67.9% | 79 | 6 |  |  | 85.1% |
| fact | idf_recall |  | 0.05 | 82.6% | 43 | 4 |  |  | 85.2% |
| fact | idf_recall |  | 0.1 | 82.6% | 43 | 4 |  |  | 85.2% |
| fact | idf_recall |  | 0.2 | 73.4% | 64 | 4 |  |  | 86.2% |
| fact | idf_recall |  | 0.25 | 69.8% | 74 | 6 |  |  | 87.3% |
| fact | idf_recall |  | 0.3 | 66.5% | 83 | 7 |  |  | 84.9% |
| fact | idf_recall |  | 0.34 | 64.9% | 86 | 7 |  |  | 85.0% |
| fact | idf_recall |  | 0.4 | 59.6% | 96 | 15 |  |  | 82.9% |
| fact | idf_recall |  | 0.5 | 47.8% | 127 | 29 |  |  | 80.2% |
| fact | tfidf_cosine |  | 0.05 | 73.5% | 64 | 6 |  |  | 87.1% |
| fact | tfidf_cosine |  | 0.1 | 52.1% | 116 | 30 |  |  | 77.5% |
| fact | tfidf_cosine |  | 0.2 | 20.3% | 179 | 73 |  |  | 64.2% |
| fact | tfidf_cosine |  | 0.25 | 11.8% | 197 | 90 |  |  | 54.4% |
| fact | tfidf_cosine |  | 0.3 | 6.5% | 207 | 99 |  |  | 45.8% |
| fact | tfidf_cosine |  | 0.34 | 2.7% | 215 | 106 |  |  | 41.2% |
| fact | tfidf_cosine |  | 0.4 | 1.3% | 218 | 109 |  |  | 39.5% |
| fact | tfidf_cosine |  | 0.5 | 0.4% | 220 | 111 |  |  | 39.1% |
| evidence | recall | lexical | 0.01 | 18.1% | 0 | 0 | 20 | 10 | 85.6% |
| evidence | recall | lexical | 0.1 | 18.1% | 0 | 0 | 20 | 10 | 85.6% |
| evidence | recall | lexical | 0.2 | 17.3% | 0 | 0 | 20 | 10 | 85.6% |
| evidence | recall | lexical | 0.25 | 15.7% | 0 | 0 | 22 | 12 | 85.2% |
| evidence | recall | lexical | 0.34 | 8.2% | 0 | 0 | 31 | 19 | 82.2% |
| evidence | recall | lexical | 0.5 | 7.7% | 0 | 0 | 32 | 20 | 81.1% |
| evidence | recall | safe | 0.01 | 21.4% | 0 | 0 | 15 | 8 | 83.5% |
| evidence | recall | safe | 0.1 | 21.4% | 0 | 0 | 15 | 8 | 83.5% |
| evidence | recall | safe | 0.2 | 20.7% | 0 | 0 | 15 | 8 | 83.5% |
| evidence | recall | safe | 0.25 | 19.5% | 0 | 0 | 17 | 10 | 83.3% |
| evidence | recall | safe | 0.34 | 13.3% | 0 | 0 | 23 | 15 | 82.2% |
| evidence | recall | safe | 0.5 | 13.0% | 0 | 0 | 23 | 15 | 82.2% |
| evidence | fuzzy_recall | lexical | 0.01 | 19.4% | 0 | 0 | 16 | 7 | 85.7% |
| evidence | fuzzy_recall | lexical | 0.1 | 19.4% | 0 | 0 | 16 | 7 | 85.7% |
| evidence | fuzzy_recall | lexical | 0.2 | 18.6% | 0 | 0 | 16 | 7 | 85.7% |
| evidence | fuzzy_recall | lexical | 0.25 | 16.8% | 0 | 0 | 18 | 9 | 85.2% |
| evidence | fuzzy_recall | lexical | 0.34 | 9.2% | 0 | 0 | 27 | 16 | 83.2% |
| evidence | fuzzy_recall | lexical | 0.5 | 8.6% | 0 | 0 | 28 | 17 | 82.5% |
| evidence | fuzzy_recall | safe | 0.01 | 22.4% | 0 | 0 | 14 | 7 | 83.8% |
| evidence | fuzzy_recall | safe | 0.1 | 22.4% | 0 | 0 | 14 | 7 | 83.8% |
| evidence | fuzzy_recall | safe | 0.2 | 21.7% | 0 | 0 | 14 | 7 | 83.8% |
| evidence | fuzzy_recall | safe | 0.25 | 20.3% | 0 | 0 | 16 | 9 | 83.2% |
| evidence | fuzzy_recall | safe | 0.34 | 13.9% | 0 | 0 | 22 | 14 | 83.0% |
| evidence | fuzzy_recall | safe | 0.5 | 13.6% | 0 | 0 | 22 | 14 | 83.0% |

## Are the current thresholds still optimal?

Current class constants (fact tau 0.3 recall, evidence tau 0.0, threshold 0.2535): dev 193/221 = 87.33%; grouped CV with these pruning settings (threshold re-tuned in the folds) 86.83%; pairs 7113, with early exit 3690.

| config | final setting | final threshold | fold thresholds min/median/max | settings chosen in the 50 folds |
|---|---|---|---|---|
| A | no pruning | 0.436 | 0.325 / 0.436 / 0.965 | no pruning (50) |
| B | fact idf_recall>=0.25 | 0.2535 | 0.032 / 0.253 / 0.331 | fact idf_recall>=0.34 (28); fact recall>=0.50 (5); fact tfidf_cosine>=0.05 (5) |
| C | evidence lexical fuzzy_recall>=0.20 | 0.408 | 0.003 / 0.288 / 0.581 | evidence lexical recall>=0.20 (21); evidence lexical recall>=0.34 (15); evidence lexical fuzzy_recall>=0.20 (11) |
| D | fact recall>=0.20, evidence safe recall>=0.20 | 0.325 | 0.003 / 0.010 / 0.649 | fact recall>=0.34, evidence safe recall>=0.25 (13); fact recall>=0.50, evidence lexical recall>=0.25 (11); fact fuzzy_recall>=0.30, evidence lexical fuzzy_recall>=0.20 (5) |

| threshold | accuracy | FP | FN | macro_f1 | pairs with early exit |
|---|---|---|---|---|---|
| 0.05 | 84.2% | 25 | 10 | 84.1% | 3035 |
| 0.1 | 83.3% | 22 | 15 | 83.2% | 3263 |
| 0.15 | 83.7% | 19 | 17 | 83.7% | 3380 |
| 0.2 | 86.0% | 13 | 18 | 86.0% | 3594 |
| 0.25 | 87.3% | 10 | 18 | 87.3% | 3690 |
| 0.2535 | 87.3% | 10 | 18 | 87.3% | 3690 |
| 0.3 | 87.3% | 9 | 19 | 87.3% | 3704 |
| 0.35 | 86.4% | 9 | 21 | 86.4% | 3767 |
| 0.4 | 86.4% | 9 | 21 | 86.4% | 3767 |
| 0.45 | 86.0% | 9 | 22 | 85.9% | 3777 |
| 0.5 | 85.5% | 9 | 23 | 85.5% | 3844 |
| 0.55 | 85.5% | 8 | 24 | 85.5% | 3908 |
| 0.6 | 85.5% | 8 | 24 | 85.5% | 3908 |
| 0.65 | 85.1% | 8 | 25 | 85.0% | 3915 |
| 0.7 | 84.6% | 8 | 26 | 84.5% | 3978 |
| 0.75 | 84.2% | 8 | 27 | 84.1% | 4045 |
| 0.8 | 82.8% | 8 | 30 | 82.7% | 4118 |
| 0.85 | 83.3% | 6 | 31 | 83.1% | 4240 |
| 0.9 | 81.9% | 5 | 35 | 81.6% | 4527 |
| 0.95 | 79.6% | 4 | 41 | 79.1% | 4744 |

## Recommended class constants (best nested grouped-CV accuracy, ties: fewer pairs): B

```python
    DEFAULT_THRESHOLD = 0.2535
    FACT_PRUNE_THRESHOLD = 0.25
    PRUNE_METRIC = "idf_recall"
    EVIDENCE_PRUNE_THRESHOLD = 0.0
    EVIDENCE_KEEP_TOP_K = 3
    EVIDENCE_KEEP_LEAD = True
```

Figures: report/current_run_figures/part2_accuracy_vs_pairs.png, report/current_run_figures/part2_threshold_accuracy.png
