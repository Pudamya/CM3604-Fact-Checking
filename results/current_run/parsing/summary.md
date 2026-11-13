# Optional parsing hybrid results

Development accuracy: **185/221 = 83.71%**.

Confusion matrix [TP FN; FP TN]: [86 26; 10 99].

Score = 0.10 x dependency recall + 0.90 x lexical recall.
Decision threshold: 0.5276785714285713.

Grouped CV, fixed weight and retuned threshold: 80.00% (SD 1.59%).
Grouped CV, weight AND threshold selected inside training folds: 80.00% (SD 1.59%).

Both validations use 10 repeats x 5 folds grouped by biography. Zero-weight lexical-only and weight-one dependency-only endpoints are comparators, not eligible hybrid configurations. The final weight is chosen by fixed-configuration grouped CV, with full-dev accuracy as a tie-break; its final threshold is selected on all development facts. Weight-selection CV reports the separate held-out behavior of the training-fold selection procedure.

The development accuracy is optimistic; this is not an unseen-test guarantee. The method is a hybrid, not pure dependency parsing. The original 151/221 parser results are preserved under results/current_run/parsing_baseline/. Graded classifiers and Part 1/2/3 results are unchanged.

Dependency recall uses labelled lemma triples. Lexical recall reuses the existing Part 1 scoring function on eligible passage sentences. Candidate sentences with incompatible explicit predicate negation or directly reversed subject/object arguments are excluded before either component is scored. Title/pronoun substitution is a limited heuristic. The lexical component can still over-credit word overlap, and the guards do not solve full semantic entailment.

Scoring time: 7.80 seconds, excluding model load. Python 3.10.11, spaCy 3.8.2, en_core_web_sm 3.8.0, Windows.
