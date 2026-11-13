"""Reproduce the submitted defaults and unpruned controls from saved NLI scores."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import factcheck as fc
import factcheck_experiments as fx


def main():
    examples = fx._x_load_examples(str(ROOT / 'data/dev_labeled_ChatGPT.jsonl'),
                                   str(ROOT / 'data/passages_bm25_ChatGPT_humfacts.jsonl'))
    cache = ROOT / 'results/current_run/part2/unit_scores.jsonl'
    records = [json.loads(line) for line in cache.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(records) == len(examples) == 221
    assert all(r['fingerprint'] == fx._x_fingerprint() for r in records), 'Cache fingerprint mismatch'
    assert all(r['index'] == i and r['fact'] == e.fact and r['label'] == e.label
               for i, (r, e) in enumerate(zip(records, examples))), 'Cache/data mismatch'
    fx._x_annotate(records, examples)
    golds = [fx._x_gold_of(e) for e in examples]
    current = fx._x_current_cfg()
    baseline = dict(current, fact_tau=0.0, ev_tau=0.0)
    settings = [
        ('before_pruning', baseline, 0.436032310128212),
        ('with_pruning', current, fc.EntailmentFactChecker.DEFAULT_THRESHOLD),
        ('unpruned_same_threshold', baseline, fc.EntailmentFactChecker.DEFAULT_THRESHOLD),
    ]
    summaries, all_predictions = [], {}
    for name, cfg, threshold in settings:
        run = fx._x_run(records, cfg)
        stats = fx._x_evaluate(run['values'], golds, threshold)
        cm = [[stats['TP'], stats['FN']], [stats['FP'], stats['TN']]]
        predictions = []
        for r, score, pruned, evidence, gold in zip(records, run['values'], run['pruned'], run['best'], golds):
            pred = 'S' if score >= threshold else 'NS'
            predictions.append({'index': r['index'], 'person': r['person'], 'fact': r['fact'],
                                'original_label': r['label'], 'gold_binary': gold, 'prediction': pred,
                                'score': score, 'pruned': pruned, 'best_evidence': evidence})
        folder = ROOT / 'experiments' / name
        folder.mkdir(exist_ok=True)
        fx._x_write_csv(folder / 'predictions.csv', predictions)
        all_predictions[name] = predictions
        summary = {'configuration': name, 'source': 'exact replay of supplied real-model probability cache',
                   'cache_fingerprint': fx._x_fingerprint(), 'facts': len(records),
                   'correct': stats['correct'], 'accuracy': stats['accuracy'],
                   'threshold': threshold, 'fact_pruning_threshold': cfg['fact_tau'],
                   'evidence_pruning_threshold': cfg['ev_tau'], 'pruning_metric': cfg['metric'],
                   'confusion_matrix_gold_rows_S_NS': cm, 'FP': cm[1][0], 'FN': cm[0][1],
                   'facts_pruned': sum(run['pruned']),
                   'supported_facts_pruned': sum(p and g == 'S' for p, g in zip(run['pruned'], golds)),
                   'pairs_before_early_exit': run['pairs'],
                   'pairs_with_early_exit_batch_16': fx._x_early_pairs(run, threshold, 16),
                   'wall_time_seconds': None, 'peak_memory_MB': None}
        (folder / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        (folder / 'README.md').write_text(
            '# ' + name.replace('_', ' ').title() + '\n\n'
            f"Accuracy: **{summary['correct']}/221 ({summary['accuracy']:.2%})**. "
            f"Confusion matrix [TP FN; FP TN]: {cm}.\n\n"
            f"Decision threshold: {threshold}; fact pruning: {cfg['fact_tau']} ({cfg['metric']}); "
            f"evidence pruning: {cfg['ev_tau']}.\n\n"
            'These are exact calculations from the supplied real-model score cache, not a new model inference run. '
            'Wall time and peak memory have not been measured in this replay.\n', encoding='utf-8')
        summaries.append(summary)
    fields = ['configuration', 'correct', 'facts', 'accuracy', 'threshold', 'fact_pruning_threshold',
              'evidence_pruning_threshold', 'FP', 'FN', 'facts_pruned', 'supported_facts_pruned',
              'pairs_before_early_exit', 'pairs_with_early_exit_batch_16']
    fx._x_write_csv(ROOT / 'experiments/pruning_results.csv', [{k: s[k] for k in fields} for s in summaries])
    changes = []
    for a, b in zip(all_predictions['before_pruning'], all_predictions['with_pruning']):
        if a['prediction'] != b['prediction']:
            changes.append({'index': a['index'], 'fact': a['fact'], 'gold': a['gold_binary'],
                            'before_pruning': a['prediction'], 'with_pruning': b['prediction'],
                            'improved': b['prediction'] == b['gold_binary']})
    fx._x_write_csv(ROOT / 'experiments/pruning_changed_predictions.csv', changes)
    for s in summaries:
        print(s['configuration'], s['correct'], s['confusion_matrix_gold_rows_S_NS'],
              s['pairs_before_early_exit'], s['pairs_with_early_exit_batch_16'])
    assert summaries[0]['correct'] == 188
    assert summaries[1]['correct'] == 193


if __name__ == '__main__':
    main()
