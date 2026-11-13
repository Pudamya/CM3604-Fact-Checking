"""Evaluate the optional dependency-and-lexical hybrid without changing graded models."""
import argparse
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import factcheck as fc
import factcheck_experiments as fx

CANDIDATE_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='results/current_run/parsing')
    args = parser.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    examples = fx._x_load_examples(str(ROOT / 'data/dev_labeled_ChatGPT.jsonl'),
                                   str(ROOT / 'data/passages_bm25_ChatGPT_humfacts.jsonl'))
    checker = fc.DependencyRecallThresholdFactChecker()
    started = time.perf_counter()
    components = [checker.score_components(e.fact, e.passages) for e in examples]
    seconds = time.perf_counter() - started
    golds = [fx._x_gold_of(e) for e in examples]
    groups = [e.passages[0].get('title', '') if e.passages else '' for e in examples]

    def scores(weight):
        return [weight * dep + (1 - weight) * lex for dep, lex in components]

    values = scores(checker.dependency_weight)
    metrics = fx._x_evaluate(values, golds, checker.threshold)
    best_threshold, _ = fx._x_select_threshold(values, golds)
    options, weight_rows = {}, []
    for weight in (0.0,) + CANDIDATE_WEIGHTS + (1.0,):
        candidate = scores(weight)
        threshold, _ = fx._x_select_threshold(candidate, golds)
        candidate_metrics = fx._x_evaluate(candidate, golds, threshold)
        cv = fx._x_grouped_cv({'x': {'values': candidate, 'pairs': 0}}, golds, groups, 10)
        weight_rows.append(dict(configuration=('guarded lexical comparator' if weight == 0 else
                                              'guarded dependency comparator' if weight == 1 else 'hybrid'),
                               dependency_weight=weight, threshold=threshold,
                               cv_accuracy=cv['cv_mean'], cv_sd=cv['cv_sd'], **candidate_metrics))
        if weight in CANDIDATE_WEIGHTS:
            options[str(weight)] = {'values': candidate, 'pairs': 0}
    fixed_cv = fx._x_grouped_cv({'x': {'values': values, 'pairs': 0}}, golds, groups, 10)
    selection_cv = fx._x_grouped_cv(options, golds, groups, 10)
    eligible = [row for row in weight_rows if row['dependency_weight'] in CANDIDATE_WEIGHTS]
    recommended = max(eligible, key=lambda row: (row['cv_accuracy'], row['accuracy'], -row['dependency_weight']))
    rows = [{'index': i, 'fact': e.fact, 'original_label': e.label, 'gold_binary': gold,
             'dependency_score': dep, 'lexical_score': lex, 'score': score,
             'prediction': 'S' if score > 0 and score >= checker.threshold else 'NS'}
            for i, (e, gold, (dep, lex), score) in enumerate(zip(examples, golds, components, values))]
    fx._x_write_csv(out / 'predictions.csv', rows)
    fx._x_write_csv(out / 'weight_comparison.csv', weight_rows)
    thresholds = sorted(set([0.05 * i for i in range(1, 21)] + [checker.threshold, best_threshold]))
    fx._x_write_csv(out / 'threshold_sweep.csv',
                    [dict(threshold=t, **fx._x_evaluate(values, golds, t)) for t in thresholds])
    summary = {'method': 'hybrid dependency recall and lexical recall with structural sentence guards',
               'python': platform.python_version(), 'platform': platform.platform(),
               'spacy_version': fc.spacy.__version__, 'parser_model': checker.MODEL_NAME,
               'parser_model_version': checker.nlp.meta.get('version'),
               'threshold': checker.threshold, 'dependency_weight': checker.dependency_weight,
               'lexical_weight': 1 - checker.dependency_weight, 'best_dev_threshold': best_threshold,
               'metrics': metrics, 'fixed_weight_grouped_cv_accuracy': fixed_cv['cv_mean'],
               'fixed_weight_grouped_cv_sd': fixed_cv['cv_sd'],
               'weight_and_threshold_grouped_cv_accuracy': selection_cv['cv_mean'],
               'weight_and_threshold_grouped_cv_sd': selection_cv['cv_sd'],
               'training_fold_weight_choices': dict(selection_cv['chosen']),
               'training_fold_threshold_min': min(selection_cv['thresholds']),
               'training_fold_threshold_max': max(selection_cv['thresholds']),
               'cv_repeats': 10, 'cv_folds': 5, 'candidate_dependency_weights': CANDIDATE_WEIGHTS,
               'recommended_dependency_weight': recommended['dependency_weight'],
               'recommended_threshold': recommended['threshold'],
               'scoring_seconds_including_parsing': seconds, 'timing_excludes_model_load': True,
               'parsed_text_cache_entries': len(checker._cache)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Optional parsing hybrid results', '',
             f"Development accuracy: **{metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}**.", '',
             f"Confusion matrix [TP FN; FP TN]: [{metrics['TP']} {metrics['FN']}; {metrics['FP']} {metrics['TN']}].", '',
             f"Score = {checker.dependency_weight:.2f} x dependency recall + {1-checker.dependency_weight:.2f} x lexical recall.",
             f"Decision threshold: {checker.threshold:.16g}.", '',
             f"Grouped CV, fixed weight and retuned threshold: {fixed_cv['cv_mean']:.2%} (SD {fixed_cv['cv_sd']:.2%}).",
             f"Grouped CV, weight AND threshold selected inside training folds: {selection_cv['cv_mean']:.2%} "
             f"(SD {selection_cv['cv_sd']:.2%}).", '',
             'Both validations use 10 repeats x 5 folds grouped by biography. Zero-weight lexical-only and '
             'weight-one dependency-only endpoints are comparators, not eligible hybrid configurations. '
             'The final weight is chosen by fixed-configuration grouped CV, with full-dev accuracy as a tie-break; '
             'its final threshold is selected on all development facts. Weight-selection CV reports the separate '
             'held-out behavior of the training-fold selection procedure.', '',
             'The development accuracy is optimistic; this is not an unseen-test guarantee. '
             'The method is a hybrid, not pure dependency parsing. The original 151/221 parser results are '
             'preserved under results/current_run/parsing_baseline/. Graded classifiers and Part 1/2/3 results are unchanged.', '',
             'Dependency recall uses labelled lemma triples. Lexical recall reuses the existing Part 1 scoring '
             'function on eligible passage sentences. Candidate sentences with incompatible explicit predicate '
             'negation or directly reversed subject/object arguments are excluded before either component is scored. '
             'Title/pronoun substitution is a limited heuristic. The lexical component can still over-credit '
             'word overlap, and the guards do not solve full semantic entailment.', '',
             f"Scoring time: {seconds:.2f} seconds, excluding model load. Python {platform.python_version()}, "
             f"spaCy {fc.spacy.__version__}, en_core_web_sm {checker.nlp.meta.get('version')}, {platform.system()}.", '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:13]))
    print('Saved:', out)


if __name__ == '__main__':
    main()
