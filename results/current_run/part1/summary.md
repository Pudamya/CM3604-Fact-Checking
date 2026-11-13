# Part 1 experiment results

Facts: 221 (112 S, 109 NS incl. IR), 32 people. AUC = ROC-AUC (threshold-free); CI = paired bootstrap 95% interval over facts (2000 resamples); delta = AUC minus the chosen configuration (recall, max over passages, chosen preprocessing). dev_acc = accuracy at the best threshold on all facts (optimistic); cv_acc = cross-validation grouped by person (10 repeats x 5 folds, threshold re-tuned on the training people).

## Final classifier (what factchecking_main.py --mode word_overlap runs)

Threshold 0.5635. Accuracy 184/221 = 83.26%. Confusion matrix [S: TP FN; NS: FP TN] = [86 26; 11 98]. S precision/recall/F1 0.887/0.768/0.823; NS 0.790/0.899/0.841. Grouped CV accuracy 80.32% +- 1.05.

## Best aggregation of each metric

| metric | aggregation | auc | auc_ci | delta_auc | delta_ci | significance | threshold | dev_acc | cv_acc | cv_acc_sd | mean_score_S | mean_score_NS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall (= ROUGE-1 recall) | max | 0.8975 | [0.850, 0.937] | 0 | [0.000, 0.000] | reference | 0.5857 | 83.3% | 80.3% | 1.1% | 0.789 | 0.267 |
| IDF-weighted recall | max | 0.8955 | [0.848, 0.936] | -0.002 | [-0.008, 0.004] | not significant | 0.342 | 83.7% | 79.2% | 1.6% | 0.758 | 0.231 |
| TF-IDF cosine | sentence_max | 0.8843 | [0.839, 0.927] | -0.0132 | [-0.045, 0.018] | not significant | 0.2378 | 81.9% | 78.0% | 1.8% | 0.351 | 0.123 |
| ROUGE-L recall | concat | 0.8828 | [0.831, 0.925] | -0.0147 | [-0.035, 0.004] | not significant | 0.4143 | 82.4% | 80.0% | 3.1% | 0.767 | 0.274 |
| Jaccard | sentence_max | 0.7987 | [0.738, 0.854] | -0.0987 | [-0.150, -0.047] | worse | 0.1716 | 72.4% | 67.4% | 1.6% | 0.203 | 0.079 |
| F1 | sentence_max | 0.7983 | [0.738, 0.854] | -0.0992 | [-0.150, -0.047] | worse | 0.2929 | 72.4% | 67.0% | 1.2% | 0.32 | 0.139 |
| precision | max | 0.7585 | [0.691, 0.817] | -0.139 | [-0.199, -0.086] | worse | 0.0147 | 72.4% | 65.0% | 2.2% | 0.034 | 0.017 |
| ROUGE-2 recall | max | 0.7436 | [0.680, 0.804] | -0.1539 | [-0.217, -0.089] | worse | 0.3542 | 70.6% | 66.7% | 1.8% | 0.356 | 0.095 |
| BLEU-4 | sentence_max | 0.6653 | [0.591, 0.734] | -0.2322 | [-0.307, -0.163] | worse | 0 | 67.4% | 62.7% | 5.6% | 0.051 | 0.023 |

## All metric / aggregation combinations

| metric | aggregation | stop_words | auc | auc_ci | delta_auc | delta_ci | significance | threshold | dev_acc | cv_acc | cv_acc_sd | mean_score_S | mean_score_NS | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall (= ROUGE-1 recall) | max | removed | 0.8975 | [0.850, 0.937] | 0 | [0.000, 0.000] | reference | 0.5857 | 83.3% | 80.3% | 1.1% | 0.789 | 0.267 | 0.091 |
| recall (= ROUGE-1 recall) | concat | removed | 0.8869 | [0.839, 0.929] | -0.0106 | [-0.024, 0.001] | not significant | 0.45 | 81.0% | 74.0% | 1.9% | 0.809 | 0.295 | 0.1 |
| recall (= ROUGE-1 recall) | sentence_max | removed | 0.8913 | [0.844, 0.932] | -0.0061 | [-0.027, 0.016] | not significant | 0.4722 | 83.7% | 82.5% | 1.7% | 0.725 | 0.239 | 0.131 |
| precision | max | removed | 0.7585 | [0.691, 0.817] | -0.139 | [-0.199, -0.086] | worse | 0.0147 | 72.4% | 65.0% | 2.2% | 0.034 | 0.017 | 0.095 |
| precision | concat | removed | 0.6793 | [0.606, 0.746] | -0.2182 | [-0.289, -0.154] | worse | 0.0037 | 67.4% | 59.2% | 1.9% | 0.012 | 0.008 | 0.107 |
| precision | sentence_max | removed | 0.7259 | [0.657, 0.790] | -0.1716 | [-0.237, -0.108] | worse | 0.2111 | 67.4% | 60.8% | 2.0% | 0.26 | 0.131 | 0.117 |
| F1 | max | removed | 0.7672 | [0.700, 0.825] | -0.1303 | [-0.188, -0.079] | worse | 0.0282 | 72.4% | 64.6% | 1.3% | 0.062 | 0.031 | 0.114 |
| F1 | concat | removed | 0.6886 | [0.615, 0.755] | -0.2089 | [-0.280, -0.145] | worse | 0.0078 | 67.9% | 58.0% | 2.2% | 0.024 | 0.015 | 0.112 |
| F1 | sentence_max | removed | 0.7983 | [0.738, 0.854] | -0.0992 | [-0.150, -0.047] | worse | 0.2929 | 72.4% | 67.0% | 1.2% | 0.32 | 0.139 | 0.135 |
| Jaccard | max | removed | 0.7672 | [0.700, 0.825] | -0.1303 | [-0.188, -0.080] | worse | 0.0143 | 72.4% | 64.6% | 1.3% | 0.033 | 0.016 | 0.092 |
| Jaccard | concat | removed | 0.6885 | [0.615, 0.755] | -0.209 | [-0.280, -0.145] | worse | 0.0039 | 67.9% | 58.0% | 2.2% | 0.012 | 0.008 | 0.13 |
| Jaccard | sentence_max | removed | 0.7987 | [0.738, 0.854] | -0.0987 | [-0.150, -0.047] | worse | 0.1716 | 72.4% | 67.4% | 1.6% | 0.203 | 0.079 | 0.121 |
| TF-IDF cosine | max | removed | 0.8596 | [0.805, 0.904] | -0.0378 | [-0.077, -0.004] | worse | 0.0931 | 80.5% | 77.4% | 0.9% | 0.173 | 0.058 | 0.259 |
| TF-IDF cosine | concat | removed | 0.8332 | [0.774, 0.883] | -0.0643 | [-0.110, -0.024] | worse | 0.0864 | 77.4% | 72.9% | 2.3% | 0.139 | 0.048 | 0.174 |
| TF-IDF cosine | sentence_max | removed | 0.8843 | [0.839, 0.927] | -0.0132 | [-0.045, 0.018] | not significant | 0.2378 | 81.9% | 78.0% | 1.8% | 0.351 | 0.123 | 0.278 |
| IDF-weighted recall | max | removed | 0.8955 | [0.848, 0.936] | -0.002 | [-0.008, 0.004] | not significant | 0.342 | 83.7% | 79.2% | 1.6% | 0.758 | 0.231 | 0.141 |
| IDF-weighted recall | concat | removed | 0.8834 | [0.835, 0.926] | -0.0141 | [-0.029, -0.000] | worse | 0.3301 | 81.9% | 74.9% | 1.9% | 0.777 | 0.257 | 0.168 |
| IDF-weighted recall | sentence_max | removed | 0.8936 | [0.846, 0.934] | -0.0039 | [-0.020, 0.013] | not significant | 0.3569 | 83.3% | 80.4% | 1.0% | 0.699 | 0.208 | 0.174 |
| ROUGE-2 recall | max | removed | 0.6483 | [0.595, 0.702] | -0.2491 | [-0.306, -0.190] | worse | 0.3095 | 65.6% | 64.7% | 1.2% | 0.246 | 0.051 | 0.205 |
| ROUGE-2 recall | concat | removed | 0.6487 | [0.595, 0.702] | -0.2488 | [-0.306, -0.190] | worse | 0.3095 | 65.6% | 64.7% | 1.2% | 0.25 | 0.051 | 0.138 |
| ROUGE-2 recall | sentence_max | removed | 0.6483 | [0.593, 0.701] | -0.2492 | [-0.306, -0.190] | worse | 0.3095 | 64.7% | 62.6% | 1.5% | 0.241 | 0.048 | 0.299 |
| ROUGE-L recall | max | removed | 0.8808 | [0.829, 0.923] | -0.0167 | [-0.034, -0.000] | worse | 0.4143 | 82.4% | 79.8% | 1.9% | 0.726 | 0.259 | 0.275 |
| ROUGE-L recall | concat | removed | 0.8828 | [0.831, 0.925] | -0.0147 | [-0.035, 0.004] | not significant | 0.4143 | 82.4% | 80.0% | 3.1% | 0.767 | 0.274 | 0.318 |
| ROUGE-L recall | sentence_max | removed | 0.8793 | [0.829, 0.923] | -0.0182 | [-0.042, 0.005] | not significant | 0.4722 | 81.9% | 80.9% | 0.5% | 0.678 | 0.233 | 0.311 |
| BLEU-4 | max | removed | 0.5729 | [0.493, 0.650] | -0.3245 | [-0.408, -0.244] | worse | 0 | 66.5% | 65.6% | 0.2% | 0.003 | 0 | 0.333 |
| BLEU-4 | concat | removed | 0.5093 | [0.425, 0.593] | -0.3882 | [-0.480, -0.299] | worse | 0 | 66.5% | 66.1% | 0.0% | 0 | 0 | 0.357 |
| BLEU-4 | sentence_max | removed | 0.6653 | [0.591, 0.734] | -0.2322 | [-0.307, -0.163] | worse | 0 | 67.4% | 62.7% | 5.6% | 0.051 | 0.023 | 0.437 |
| ROUGE-2 recall | max | kept | 0.7436 | [0.680, 0.804] | -0.1539 | [-0.217, -0.089] | worse | 0.3542 | 70.6% | 66.7% | 1.8% | 0.356 | 0.095 | 0.452 |
| ROUGE-L recall | max | kept | 0.8171 | [0.756, 0.870] | -0.0804 | [-0.125, -0.037] | worse | 0.6202 | 75.6% | 71.3% | 0.8% | 0.709 | 0.459 | 0.417 |
| BLEU-4 | max | kept | 0.4418 | [0.367, 0.518] | -0.4557 | [-0.548, -0.369] | worse | 0.0003 | 51.1% | 36.8% | 2.5% | 0.002 | 0 | 0.375 |

## Preprocessing ablation (recall; one setting changed at a time)

| change | auc | auc_ci | delta_auc | delta_ci | significance | threshold | dev_acc | cv_acc | cv_acc_sd | mean_score_S | mean_score_NS | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (chosen configuration) | 0.8975 | [0.850, 0.937] | 0 | [0.000, 0.000] | reference | 0.5857 | 83.3% | 80.3% | 1.1% | 0.789 | 0.267 | 0.081 |
| stemming: none | 0.8718 | [0.821, 0.914] | -0.0257 | [-0.045, -0.009] | worse | 0.3667 | 80.1% | 77.4% | 1.9% | 0.737 | 0.257 | 0.512 |
| stemming: Snowball | 0.8975 | [0.850, 0.937] | 0 | [0.000, 0.000] | not significant | 0.5857 | 83.3% | 80.3% | 1.1% | 0.789 | 0.267 | 0.379 |
| stemming: spaCy lemmatiser | 0.8861 | [0.837, 0.927] | -0.0113 | [-0.029, 0.001] | not significant | 0.45 | 81.9% | 77.2% | 3.4% | 0.774 | 0.268 | 14.06 |
| stop words kept | 0.8861 | [0.838, 0.929] | -0.0113 | [-0.038, 0.016] | not significant | 0.6515 | 84.2% | 84.2% | 0.0% | 0.831 | 0.508 | 0.094 |
| person's name kept | 0.8794 | [0.832, 0.922] | -0.0181 | [-0.042, 0.006] | not significant | 0.6333 | 82.4% | 80.5% | 1.3% | 0.82 | 0.407 | 0.066 |
| case kept | 0.895 | [0.848, 0.936] | -0.0025 | [-0.016, 0.009] | not significant | 0.5857 | 83.3% | 80.0% | 1.4% | 0.783 | 0.254 | 0.506 |
| min token length 1 | 0.8953 | [0.847, 0.936] | -0.0022 | [-0.007, 0.000] | not significant | 0.45 | 82.8% | 80.2% | 1.1% | 0.789 | 0.269 | 0.45 |
| aggregation: concatenated passages | 0.8869 | [0.839, 0.929] | -0.0106 | [-0.024, 0.001] | not significant | 0.45 | 81.0% | 74.0% | 1.9% | 0.809 | 0.295 | 0.09 |
| aggregation: best sentence | 0.8913 | [0.844, 0.932] | -0.0061 | [-0.027, 0.016] | not significant | 0.4722 | 83.7% | 82.5% | 1.7% | 0.725 | 0.239 | 0.116 |

## Threshold sweep of the final classifier

| threshold | accuracy | FP | FN | S_f1 | NS_f1 |
|---|---|---|---|---|---|
| 0.1 | 66.5% | 70 | 4 | 74.5% | 51.3% |
| 0.15 | 67.0% | 69 | 4 | 74.7% | 52.3% |
| 0.2 | 67.9% | 67 | 4 | 75.3% | 54.2% |
| 0.25 | 71.9% | 58 | 4 | 77.7% | 62.2% |
| 0.3 | 77.8% | 43 | 6 | 81.2% | 72.9% |
| 0.35 | 81.4% | 34 | 7 | 83.7% | 78.5% |
| 0.4 | 81.4% | 34 | 7 | 83.7% | 78.5% |
| 0.45 | 82.8% | 29 | 9 | 84.4% | 80.8% |
| 0.5 | 82.8% | 29 | 9 | 84.4% | 80.8% |
| 0.55 | 82.8% | 12 | 26 | 81.9% | 83.6% |
| 0.5635 | 83.3% | 11 | 26 | 82.3% | 84.1% |
| 0.6 | 83.3% | 10 | 27 | 82.1% | 84.3% |
| 0.65 | 81.9% | 9 | 31 | 80.2% | 83.3% |
| 0.7 | 78.3% | 7 | 41 | 74.7% | 81.0% |
| 0.75 | 78.3% | 7 | 41 | 74.7% | 81.0% |
| 0.8 | 76.0% | 5 | 48 | 70.7% | 79.7% |
| 0.85 | 74.7% | 5 | 51 | 68.5% | 78.8% |
| 0.9 | 74.2% | 5 | 52 | 67.8% | 78.5% |
| 0.95 | 74.2% | 5 | 52 | 67.8% | 78.5% |

## Score distribution of the final classifier

| score bin | S | NS |
|---|---|---|
| [0.0, 0.1) | 4 | 39 |
| [0.1, 0.2) | 0 | 3 |
| [0.2, 0.3) | 2 | 24 |
| [0.3, 0.4) | 1 | 9 |
| [0.4, 0.5) | 2 | 5 |
| [0.5, 0.6) | 18 | 19 |
| [0.6, 0.7) | 14 | 3 |
| [0.7, 0.8) | 7 | 2 |
| [0.8, 0.9) | 4 | 0 |
| [0.9, 1.0] | 60 | 5 |

Figures: report/current_run_figures/part1_roc_metrics.png, report/current_run_figures/part1_auc_by_metric.png, report/current_run_figures/part1_threshold_accuracy.png
