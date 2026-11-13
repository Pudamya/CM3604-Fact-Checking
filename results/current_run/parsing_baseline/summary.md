# Optional parsing classifier results

Accuracy: **151/221 = 68.33%**.

Confusion matrix [TP FN; FP TN]: [74 38; 32 77].

Default dependency-recall threshold: 0.236111111111. Best full-dev threshold: 0.236111111111.

Grouped cross-validation: 62.76% (SD 1.64%), 10 repeats x 5 folds grouped by person.
Only the decision threshold is retuned on training people in each fold; held-out people are evaluated.

Full-dev threshold selection is optimistic. This optional baseline is separate from the graded word-overlap and DeBERTa classifiers and has no separate accuracy requirement in the specification.

Scoring time: 4.91 seconds, excluding parser-model load; Python 3.10.11, spaCy 3.8.2, en_core_web_sm 3.8.0 on Windows.

The score is the maximum sentence-level recall of labelled dependency triples (head lemma, relation, dependent lemma). Page-title name tokens and singular personal pronouns share an entity marker. A predicate-level negation mismatch rejects the candidate sentence. Facts with no usable relations fall back to content-lemma recall. Parsing and lemmatization are cached across shared passages.

Limitations: exact dependency matching misses paraphrases and active/passive alternations. Title/pronoun substitution is a heuristic, not full coreference resolution. Negation checks use predicate lemmas rather than a full semantic scope model. Different parser-model versions may change results.
