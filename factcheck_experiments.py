import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from factcheck import (
    AGGREGATIONS,
    EntailmentFactChecker,
    EntailmentModel,
    TextPreprocessor,
    WordRecallThresholdFactChecker,
    bleu,
    fuzzy_recall,
    overlap_recall,
    rouge_l_recall,
    rouge_n_recall,
    word_overlap_score
)


# --- PART 2: EXPERIMENT COMMAND ---
_EXPERIMENTS_DOC = """
Part 2 experiments: cache DeBERTa scores, compare pruning methods,
and tune settings using cross-validation grouped by person.
Report accuracy, errors and scoring costs.
Use --timing to measure runtime and memory use.
"""
_x_ROOT = Path(__file__).resolve().parent
_x_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # the checkpoint loaded by factchecking_main.py
_x_FACT_TAUS = (0.05, 0.1, 0.2, 0.25, 0.3, 0.34, 0.4, 0.5)
_x_EVID_TAUS = (0.01, 0.1, 0.2, 0.25, 0.34, 0.5)
_x_FACT_METRICS = ("recall", "fuzzy_recall", "idf_recall", "tfidf_cosine")  # fact-level pruning 
_x_UNIT_METRICS = ("recall", "fuzzy_recall")  # metrics that work for evidence-level pruning 
_x_EVID_RULES = {"lexical": (0, False), "safe": (3, True)}  
_x_BASE = {"mode": "context", "titled": False, "min_words": 0, "metric": "recall", "fact_tau": 0.0, "ev_tau": 0.0,
           "rule": "safe", "strategy": "max_entailment"}
_x_PREVIOUS = dict(_x_BASE, name="previous submission (context, no pruning, t=0.436)", threshold=0.436032310128212)
_x_FAMILIES = (("A", "no pruning (baseline)"), ("B", "fact-level pruning only"), ("C", "evidence-level pruning only"),
               ("D", "fact-level + evidence-level pruning"))


def _x_load_examples(labels_path, passages_path):
    """ Uses the unmodified readers of factchecking_main.py, so the facts and their order match the driver. """
    import factchecking_main as fm
    return fm.read_fact_examples(labels_path, fm.read_passages(passages_path))


def _x_gold_of(example):
    return "NS" if example.label == "IR" else example.label


def _x_load_model(cuda):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(_x_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(_x_MODEL_NAME)
    if cuda:
        model.to("cuda")
    return model, tokenizer


def _x_fingerprint():
    """Hash evidence and inference source to validate cached scores."""
    parts = [_x_MODEL_NAME]
    for obj in (EntailmentModel._windows, EntailmentModel.predict_probs, EntailmentFactChecker._clean_evidence,
                EntailmentFactChecker._sentence_units, EntailmentFactChecker._entity_name,
                EntailmentFactChecker._resolve_pronoun, EntailmentFactChecker.evidence_records):
        parts.append(inspect.getsource(obj))
    parts.append(EntailmentFactChecker._PRONOUN_RE.pattern)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _x_score_units(examples, cache_path, cuda, batch_size):
    """ Scores every candidate unit once with the real model; resumable (one JSON line per fact). """
    fp = _x_fingerprint()
    cached = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("fingerprint") == fp:
                    cached[row["index"]] = row
        if cached:
            print("Reusing", len(cached), "cached facts from", cache_path)
    todo = [i for i, e in enumerate(examples) if i not in cached or cached[i]["fact"] != e.fact]
    if todo:
        model, tokenizer = _x_load_model(cuda)
        ent_model = EntailmentModel(model, tokenizer, cuda, batch_size=batch_size)
        checker = EntailmentFactChecker(ent_model)
        started = time.perf_counter()
        with cache_path.open("a" if cached else "w", encoding="utf-8") as stream:
            for n, i in enumerate(todo, 1):
                e = examples[i]
                units = checker.evidence_records(e.passages, "context", titled=True) + \
                    checker.evidence_records(e.passages, "passage")
                texts = [u["text"] for u in units]
                probs = ent_model.predict_probs(texts, e.fact) if texts else []
                hyp = tokenizer.encode(e.fact, add_special_tokens=False, verbose=False)[:ent_model.max_length // 2]
                ids = tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"] if texts else []
                rows = [{"text": u["text"], "kind": u["kind"], "lead": u["lead"], "tokens": len(t),
                         "windows": len(ent_model._windows(t, hyp)) if t else 0, "p": p}
                        for u, t, p in zip(units, ids, probs)]
                record = {"fingerprint": fp, "index": i, "fact": e.fact, "label": e.label, "gold": _x_gold_of(e),
                          "person": e.passages[0]["title"] if e.passages else "", "units": rows}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                cached[i] = record
                elapsed = time.perf_counter() - started
                print("\rscored %d/%d facts (%.1f s/fact)" % (n, len(todo), elapsed / n), end="", flush=True)
        print()
    return [cached[i] for i in range(len(examples))]


def _x_annotate(records, examples):
    """ Adds the word overlaps used by pruning (computed with the classifier's own functions, both metrics). """
    checker = EntailmentFactChecker(None)
    for record, e in zip(records, examples):
        record["fact_overlap"] = {m: checker.word_overlap(e.fact, e.passages, m) for m in _x_FACT_METRICS}
        for u in record["units"]:
            u["overlap"] = {m: checker.unit_overlap(e.fact, e.passages, u["text"], m) for m in _x_UNIT_METRICS}


def _x_current_cfg():
    """ The configuration given by the EntailmentFactChecker class constants (what factchecking_main.py runs). """
    C = EntailmentFactChecker
    rule = "safe" if (C.EVIDENCE_KEEP_TOP_K, C.EVIDENCE_KEEP_LEAD) == _x_EVID_RULES["safe"] else "lexical"
    return dict(_x_BASE, metric=C.PRUNE_METRIC, fact_tau=C.FACT_PRUNE_THRESHOLD, ev_tau=C.EVIDENCE_PRUNE_THRESHOLD,
                rule=rule, top_k=C.EVIDENCE_KEEP_TOP_K, lead=C.EVIDENCE_KEEP_LEAD, mode=C.EVIDENCE_MODE,
                titled=C.TITLED_PRONOUN_UNITS, min_words=C.MIN_EVIDENCE_WORDS)


def _x_kinds(cfg):
    if cfg["mode"] == "passage":
        return {"passage"}
    kinds = {"sentence", "context"} if cfg["mode"] == "context" else {"sentence"}
    return kinds | ({"titled"} if cfg["titled"] else set())


def _x_run(records, cfg):
    """Evaluate a configuration exactly from cached evidence probabilities."""
    top_k, lead = cfg.get("top_k"), cfg.get("lead")
    if top_k is None or lead is None:
        top_k, lead = _x_EVID_RULES[cfg["rule"]]
    kinds, metric = _x_kinds(cfg), cfg["metric"]
    unit_metric = metric if metric in _x_UNIT_METRICS else "recall"  # as EntailmentFactChecker.unit_overlap()
    out = {"values": [], "best": [], "pruned": [], "orders": [], "pairs": 0, "inputs": 0, "tokens": 0, "cfg": cfg}
    for r in records:
        if cfg["fact_tau"] > 0 and r["fact_overlap"][metric] < cfg["fact_tau"]:
            out["values"].append(0.0 if cfg["strategy"] != "entailment_minus_contradiction" else -1.0)
            out["best"].append("")
            out["pruned"].append(True)
            out["orders"].append([])
            continue
        units = [u for u in r["units"] if u["kind"] in kinds]
        plan = EntailmentFactChecker.plan_evidence(units, [u["overlap"][unit_metric] for u in units], cfg["min_words"],
                                                   cfg["ev_tau"], top_k, lead)
        kept = [units[i] for i in plan]
        out["pairs"] += len(kept)
        out["inputs"] += sum(u["windows"] for u in kept)
        out["tokens"] += sum(u["tokens"] for u in kept)
        out["pruned"].append(False)
        out["orders"].append([u["p"][0] for u in kept])
        if not kept:
            out["values"].append(0.0 if cfg["strategy"] != "entailment_minus_contradiction" else -1.0)
            out["best"].append("")
            continue
        if cfg["strategy"] == "max_entailment":
            best = max(kept, key=lambda u: u["p"][0])
            value = best["p"][0]
        elif cfg["strategy"] == "entailment_minus_contradiction":
            best = max(kept, key=lambda u: u["p"][0] - u["p"][2])
            value = best["p"][0] - best["p"][2]
        else:  # discrete: entailed if any unit's most likely class is entailment
            hits = [u for u in kept if u["p"][0] >= max(u["p"][1], u["p"][2])]
            best = max(hits or kept, key=lambda u: u["p"][0])
            value = 1.0 if hits else 0.0
        out["values"].append(value)
        out["best"].append(best["text"])
    return out


def _x_early_pairs(run, threshold, batch=16):
    """ Pairs scored by predict() with EARLY_EXIT: batches in scoring order until one unit reaches the threshold. """
    total = 0
    for order in run["orders"]:
        hit = next((j for j, p in enumerate(order) if p >= threshold), None)
        total += len(order) if hit is None else min(len(order), (hit // batch + 1) * batch)
    return total


def _x_evaluate(values, golds, threshold):
    cm = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for v, g in zip(values, golds):
        pred = "S" if v >= threshold else "NS"
        cm[("T" if pred == g else "F") + ("P" if pred == "S" else "N")] += 1
    return _x_metrics(cm)


def _x_metrics(cm):
    n = cm["TP"] + cm["FN"] + cm["FP"] + cm["TN"]
    out = dict(cm, accuracy=(cm["TP"] + cm["TN"]) / n if n else 0.0, correct=cm["TP"] + cm["TN"], total=n)
    for label, tp, fp, fn in (("S", cm["TP"], cm["FP"], cm["FN"]), ("NS", cm["TN"], cm["FN"], cm["FP"])):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[label + "_precision"], out[label + "_recall"] = p, r
        out[label + "_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    out["macro_f1"] = (out["S_f1"] + out["NS_f1"]) / 2
    return out


def _x_select_threshold(values, golds):
    """Maximise accuracy; break ties by the widest gap between fact scores."""
    xs = sorted(set(values))
    count = {x: [0, 0] for x in xs}
    for v, g in zip(values, golds):
        count[v][0 if g == "S" else 1] += 1
    total_s = sum(c[0] for c in count.values())
    best, s_below, ns_below = None, 0, 0
    for i in range(len(xs) + 1):
        correct = ns_below + (total_s - s_below)
        if 0 < i < len(xs):
            threshold, margin = (xs[i - 1] + xs[i]) / 2, (xs[i] - xs[i - 1]) / 2
        else:
            threshold, margin = (xs[0] if i == 0 else xs[-1] + 1e-6), 0.0
        if best is None or (correct, margin) > best[0]:
            best = ((correct, margin), threshold)
        if i < len(xs):
            s_below += count[xs[i]][0]
            ns_below += count[xs[i]][1]
    return best[1], best[0][0] / len(values)


def _x_folds(groups, k, seed):
    people = sorted(set(groups))
    random.Random(seed).shuffle(people)
    fold_of = {p: i % k for i, p in enumerate(people)}
    return [fold_of[g] for g in groups]


def _x_grouped_cv(options, golds, groups, repeats, k=5):
    """Tune settings and thresholds inside training folds grouped by biography."""
    n = len(golds)
    accs, pooled, chosen, thresholds = [], Counter(), Counter(), []
    for seed in range(repeats):
        fold = _x_folds(groups, k, seed)
        correct = 0
        for f in range(k):
            train = [i for i in range(n) if fold[i] != f]
            test = [i for i in range(n) if fold[i] == f]
            best = None
            for name, run in options.items():
                t, a = _x_select_threshold([run["values"][i] for i in train], [golds[i] for i in train])
                key = (a, -run["pairs"])
                if best is None or key > best[0]:
                    best = (key, name, t)
            _, name, t = best
            chosen[name] += 1
            thresholds.append(t)
            for i in test:
                pred = "S" if options[name]["values"][i] >= t else "NS"
                pooled[("T" if pred == golds[i] else "F") + ("P" if pred == "S" else "N")] += 1
                correct += pred == golds[i]
        accs.append(correct / n)
    m = _x_metrics({key: pooled[key] for key in ("TP", "FN", "FP", "TN")})
    m.update({"cv_mean": statistics.mean(accs), "cv_sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
              "chosen": chosen, "thresholds": thresholds, "folds": repeats * k})
    return m


def _x_cv_score(run, golds, groups, repeats):
    """ Grouped CV accuracy of one fixed option (only the threshold re-tuned in the folds). """
    return _x_grouped_cv({"x": run}, golds, groups, repeats)["cv_mean"]


def _x_cfg_name(cfg):
    parts = []
    if cfg["fact_tau"] > 0:
        parts.append("fact %s>=%.2f" % (cfg["metric"], cfg["fact_tau"]))
    if cfg["ev_tau"] > 0:
        parts.append("evidence %s %s>=%.2f" % (cfg["rule"], cfg["metric"], cfg["ev_tau"]))
    return ", ".join(parts) or "no pruning"


def _x_family_options(family):
    """ Configurations (pruning settings) searched for one family. """
    cfgs = []
    if family == "A":
        cfgs.append(dict(_x_BASE))
    for metric in (_x_FACT_METRICS if family == "B" else _x_UNIT_METRICS):
        for rule in _x_EVID_RULES:
            fact_taus = _x_FACT_TAUS if family in "BD" else (0.0,)
            evid_taus = _x_EVID_TAUS if family in "CD" else (0.0,)
            if family == "B" and rule != "safe":
                continue  
            for ft in fact_taus:
                for et in evid_taus:
                    if family != "A":
                        cfgs.append(dict(_x_BASE, metric=metric, rule=rule, fact_tau=ft, ev_tau=et))
    return cfgs


def _x_peak_rss_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2 ** 20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10


def _x_timing_worker(args):
    """ Runs the real predict() for one configuration on the first N facts and prints a JSON result line. """
    cfg = json.loads(args.timing_worker)
    examples = _x_load_examples(args.labels_path, args.passages_path)[:cfg["facts"]]
    model, tokenizer = _x_load_model(args.cuda)
    if args.cuda:
        torch.cuda.reset_peak_memory_stats()
    after_load = _x_peak_rss_mb()
    ent_model = EntailmentModel(model, tokenizer, args.cuda, batch_size=cfg["batch_size"])
    checker = EntailmentFactChecker(ent_model, **cfg["checker"])
    cm = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    predictions = []
    started = time.perf_counter()
    for e in examples:
        pred = checker.predict(e.fact, e.passages)
        gold = _x_gold_of(e)
        cm[("T" if pred == gold else "F") + ("P" if pred == "S" else "N")] += 1
        predictions.append(pred)
    elapsed = time.perf_counter() - started
    result = {"name": cfg["name"], "facts": len(examples), "batch_size": cfg["batch_size"],
              "seconds": elapsed, "seconds_per_fact": elapsed / len(examples),
              "pairs": ent_model.premises_scored, "model_inputs": ent_model.pairs_scored,
              "forward_passes": ent_model.forward_passes, "peak_rss_after_load_mb": after_load,
              "peak_rss_mb": _x_peak_rss_mb(), "inference_overhead_mb": _x_peak_rss_mb() - after_load,
              "cuda_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if args.cuda else None,
              "torch_threads": torch.get_num_threads(), "predictions": predictions, **_x_metrics(cm)}
    print("TIMING_RESULT " + json.dumps(result))


def _x_run_timing(args, jobs):
    """ jobs: list of {name, batch_size, facts, checker kwargs, expected predictions or None} """
    results = []
    for job in jobs:
        print("timing:", job["name"], "on", job["facts"], "facts ...", flush=True)
        cfg = {k: v for k, v in job.items() if k != "expected"}
        cmd = [sys.executable, os.path.abspath(__file__), "experiments", "--timing-worker", json.dumps(cfg),
               "--labels_path", args.labels_path, "--passages_path", args.passages_path]
        if args.cuda:
            cmd.append("--cuda")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_x_ROOT))
        lines = [l for l in proc.stdout.splitlines() if l.startswith("TIMING_RESULT ")]
        if proc.returncode != 0 or not lines:
            print(proc.stderr[-3000:])
            raise RuntimeError("timing worker failed for " + job["name"])
        r = json.loads(lines[-1][len("TIMING_RESULT "):])
        expected = job.get("expected")
        r["matches_offline_analysis"] = None if expected is None else r["predictions"] == expected[:r["facts"]]
        del r["predictions"]
        results.append(r)
        print("  %.1f s (%.2f s/fact), %d pairs, %d forward passes, peak RSS %.0f MB, accuracy %.4f" % (
            r["seconds"], r["seconds_per_fact"], r["pairs"], r["forward_passes"], r["peak_rss_mb"], r["accuracy"]),
            flush=True)
    return results


def _x_write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _x_fmt(v, column=""):
    if isinstance(v, bool) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, float):
        if any(w in column for w in ("acc", "saved", "precision", "recall", "f1", "share", "kept")):
            return "%.1f%%" % (100 * v)
        return "%.4g" % v
    return str(v)


def _x_md_table(rows, columns, headers=None):
    lines = ["| " + " | ".join(headers or columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_x_fmt(r.get(c, ""), c) for c in columns) + " |")
    return "\n".join(lines)


def _x_figures(fig_dir, cost_rows, finals, sweep_runs, golds, timing_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: figures skipped (optional).")
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    colours = {"A": "#444444", "B": "#1f77b4", "C": "#ff7f0e", "D": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for fam in "BCD":
        pts = [r for r in cost_rows if r["family"] == fam]
        ax.scatter([r["pairs"] for r in pts], [100 * r["cv_acc"] for r in pts], s=10, alpha=0.35, color=colours[fam],
                   label=fam + ": all settings")
    for fam, f in finals.items():
        ax.scatter([f["pairs"]], [100 * f["nested"]["cv_mean"]], s=70, marker="*", color=colours[fam],
                   edgecolor="black", zorder=3)
        ax.annotate(fam, (f["pairs"], 100 * f["nested"]["cv_mean"]), textcoords="offset points", xytext=(6, 4))
    ax.axhline(83, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("premise/hypothesis pairs sent to DeBERTa (221 facts)")
    ax.set_ylabel("grouped CV accuracy (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "part2_accuracy_vs_pairs.png", dpi=200)
    plt.close(fig)
    made.append("part2_accuracy_vs_pairs.png")

    if timing_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for r in timing_rows:
            ax.scatter([r["seconds"]], [100 * r["accuracy"]], s=50)
            ax.annotate(r["name"], (r["seconds"], 100 * r["accuracy"]), textcoords="offset points", xytext=(5, 3),
                        fontsize=7)
        ax.set_xlabel("measured wall time (s)")
        ax.set_ylabel("dev accuracy of the real run (%)")
        fig.tight_layout()
        fig.savefig(fig_dir / "part2_accuracy_vs_time.png", dpi=200)
        plt.close(fig)
        made.append("part2_accuracy_vs_time.png")

    grid = [i / 100 for i in range(5, 96)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for fam, run in sweep_runs.items():
        ax.plot(grid, [100 * _x_evaluate(run["values"], golds, t)["accuracy"] for t in grid], label=fam,
                color=colours[fam])
    ax.axhline(83, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("decision threshold on max P(entailment)")
    ax.set_ylabel("dev accuracy (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "part2_threshold_accuracy.png", dpi=200)
    plt.close(fig)
    made.append("part2_threshold_accuracy.png")
    return made


def _x_main(argv):
    parser = argparse.ArgumentParser(prog="factcheck_experiments.py experiments", description=_EXPERIMENTS_DOC,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--out_dir", default="results/current_run/part2")
    parser.add_argument("--fig_dir", default="report/current_run_figures")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size of the scoring pass")
    parser.add_argument("--cv_repeats", type=int, default=10, help="repeats of the grouped 5-fold CV")
    parser.add_argument("--timing", action="store_true", help="also measure time and memory with the real model")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--sweep_facts", type=int, default=30, help="facts used by the batch-size sweep")
    parser.add_argument("--timing_facts", type=int, default=221, help="facts used by the configuration timing")
    parser.add_argument("--timing-worker", dest="timing_worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.chdir(_x_ROOT)
    if args.timing_worker:
        _x_timing_worker(args)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = _x_load_examples(args.labels_path, args.passages_path)
    print("Read", len(examples), "examples")
    records = _x_score_units(examples, out_dir / "unit_scores.jsonl", args.cuda, args.batch_size)
    print("Computing word overlaps ...", flush=True)
    _x_annotate(records, examples)
    golds = [r["gold"] for r in records]
    groups = [r["person"] for r in records]
    R = args.cv_repeats
    C = EntailmentFactChecker
    batch = EntailmentModel.BATCH_SIZE

    print("Evaluating pruning configurations ...", flush=True)
    finals, cost_rows = {}, []
    baseline_run = _x_run(records, dict(_x_BASE))
    full_pairs = baseline_run["pairs"]
    for fam, title in _x_FAMILIES:
        options = {}
        for cfg in _x_family_options(fam):
            run = _x_run(records, cfg)
            options[_x_cfg_name(cfg)] = run
        scored = []
        for name, run in options.items():
            cv = _x_cv_score(run, golds, groups, R)
            scored.append((cv, -run["pairs"], name))
            cost_rows.append({"family": fam, "setting": name, "cv_acc": cv, "pairs": run["pairs"],
                              "pairs_saved": 1 - run["pairs"] / full_pairs})
        nested = _x_grouped_cv(options, golds, groups, R)
        cv, _, name = max(scored)
        run = options[name]
        t, _ = _x_select_threshold(run["values"], golds)
        finals[fam] = {"title": title, "name": name, "cfg": run["cfg"], "run": run, "threshold": t,
                       "cv_fixed": cv, "nested": nested, "dev": _x_evaluate(run["values"], golds, t),
                       "pairs": run["pairs"], "early_pairs": _x_early_pairs(run, t, batch)}
        print("  %s %-38s nested CV %.2f%%  final: %s, t=%.4f" % (fam, title, 100 * nested["cv_mean"], name, t))

    # current class constants 
    current_cfg = _x_current_cfg()
    current_run = _x_run(records, current_cfg)
    current = {"dev": _x_evaluate(current_run["values"], golds, C.DEFAULT_THRESHOLD),
               "cv_retuned": _x_cv_score(current_run, golds, groups, R),
               "pairs": current_run["pairs"], "early_pairs": _x_early_pairs(current_run, C.DEFAULT_THRESHOLD, batch)}
    previous_run = _x_run(records, _x_PREVIOUS)
    previous = {"dev": _x_evaluate(previous_run["values"], golds, _x_PREVIOUS["threshold"]),
                "cv": _x_cv_score(previous_run, golds, groups, R), "pairs": previous_run["pairs"]}

    main_rows = []
    for fam, f in finals.items():
        d, nst = f["dev"], f["nested"]
        main_rows.append({
            "config": fam + ": " + f["title"], "final setting": f["name"], "threshold": round(f["threshold"], 4),
            "cv_acc (nested)": nst["cv_mean"], "cv_acc_sd": nst["cv_sd"], "cv_acc (final setting fixed)": f["cv_fixed"], "cv S_precision": nst["S_precision"],
            "cv S_recall": nst["S_recall"], "cv S_f1": nst["S_f1"], "cv NS_f1": nst["NS_f1"],
            "cv macro_f1": nst["macro_f1"],
            "cv CM [TP FN; FP TN] per repeat": "[%d %d; %d %d]" % tuple(round(nst[k] / R) for k in
                                                                       ("TP", "FN", "FP", "TN")),
            "dev_acc": d["accuracy"], "dev S_precision": d["S_precision"], "dev S_recall": d["S_recall"],
            "dev S_f1": d["S_f1"], "dev NS_precision": d["NS_precision"], "dev NS_recall": d["NS_recall"],
            "dev NS_f1": d["NS_f1"], "dev macro_f1": d["macro_f1"],
            "dev CM [TP FN; FP TN]": "[%d %d; %d %d]" % (d["TP"], d["FN"], d["FP"], d["TN"]),
            "FP": d["FP"], "FN": d["FN"], "facts pruned": sum(f["run"]["pruned"]),
            "S facts pruned": sum(1 for p, g in zip(f["run"]["pruned"], golds) if p and g == "S"),
            "pairs": f["pairs"], "pairs_saved": 1 - f["pairs"] / full_pairs, "pairs with early exit":
                f["early_pairs"], "early_saved": 1 - f["early_pairs"] / full_pairs,
            "model inputs (windows)": f["run"]["inputs"], "premise tokens": f["run"]["tokens"]})
    _x_write_csv(out_dir / "pruning_comparison.csv", main_rows)
    _x_write_csv(out_dir / "accuracy_vs_cost_all_settings.csv",
                 sorted(cost_rows, key=lambda r: (r["family"], -r["cv_acc"], r["pairs"])))

    selection_rows = []
    for mode, titled, label in (("sentence", False, "sentence"), ("sentence", True, "sentence + pronoun-resolved"),
                                ("context", False, "sentence + previous-sentence context"),
                                ("context", True, "context + pronoun-resolved"), ("passage", False, "whole passage")):
        run = _x_run(records, dict(_x_BASE, mode=mode, titled=titled))
        t, _ = _x_select_threshold(run["values"], golds)
        d = _x_evaluate(run["values"], golds, t)
        selection_rows.append({"evidence units": label, "threshold": round(t, 4), "dev_acc": d["accuracy"],
                               "cv_acc": _x_cv_score(run, golds, groups, R), "FP": d["FP"], "FN": d["FN"],
                               "pairs": run["pairs"], "model inputs": run["inputs"]})
    for strategy in ("entailment_minus_contradiction", "argmax_any"):
        run = _x_run(records, dict(_x_BASE, strategy=strategy))
        if strategy == "argmax_any":
            t, cvs = 0.5, _x_evaluate(run["values"], golds, 0.5)["accuracy"]
        else:
            t, cvs = _x_select_threshold(run["values"], golds)[0], _x_cv_score(run, golds, groups, R)
        d = _x_evaluate(run["values"], golds, t)
        selection_rows.append({"evidence units": "context, decision = " + strategy, "threshold": round(t, 4),
                               "dev_acc": d["accuracy"], "cv_acc": cvs, "FP": d["FP"], "FN": d["FN"],
                               "pairs": run["pairs"], "model inputs": run["inputs"]})
    _x_write_csv(out_dir / "evidence_selection.csv", selection_rows)

    base_best = baseline_run["best"]
    t_base, _ = _x_select_threshold(baseline_run["values"], golds)
    safety_rows = []
    for metric in _x_FACT_METRICS:
        for tau in _x_FACT_TAUS:
            run = _x_run(records, dict(_x_BASE, metric=metric, fact_tau=tau))
            safety_rows.append({"level": "fact", "metric": metric, "rule": "", "tau": tau,
                                "units kept": run["pairs"] / full_pairs, "facts pruned": sum(run["pruned"]),
                                "S facts pruned": sum(1 for p, g in zip(run["pruned"], golds) if p and g == "S"),
                                "best evidence lost (S-predicted facts)": "", "of which gold S": "",
                                "cv_acc": _x_cv_score(run, golds, groups, R)})
    for metric in _x_UNIT_METRICS:
        for rule in _x_EVID_RULES:
            for tau in _x_EVID_TAUS:
                run = _x_run(records, dict(_x_BASE, metric=metric, rule=rule, ev_tau=tau))
                lost = [i for i, (v0, b0, b) in enumerate(zip(baseline_run["values"], base_best, run["best"]))
                        if v0 >= t_base and b0 and b0 != b and run["values"][i] < v0]
                safety_rows.append({"level": "evidence", "metric": metric, "rule": rule, "tau": tau,
                                    "units kept": run["pairs"] / full_pairs, "facts pruned": 0, "S facts pruned": 0,
                                    "best evidence lost (S-predicted facts)": len(lost),
                                    "of which gold S": sum(1 for i in lost if golds[i] == "S"),
                                    "cv_acc": _x_cv_score(run, golds, groups, R)})
    _x_write_csv(out_dir / "pruning_safety.csv", safety_rows)

    stability_rows = []
    for fam, f in finals.items():
        nst = f["nested"]
        ts = sorted(nst["thresholds"])
        top = nst["chosen"].most_common(3)
        stability_rows.append({"config": fam, "final setting": f["name"], "final threshold": round(f["threshold"], 4),
                               "fold thresholds min/median/max": "%.3f / %.3f / %.3f" % (
                                   ts[0], ts[len(ts) // 2], ts[-1]),
                               "settings chosen in the %d folds" % nst["folds"]:
                                   "; ".join("%s (%d)" % (n, c) for n, c in top)})
    _x_write_csv(out_dir / "threshold_stability.csv", stability_rows)
    sweep_rows = []
    for t in [i / 20 for i in range(1, 20)] + [C.DEFAULT_THRESHOLD]:
        e = _x_evaluate(current_run["values"], golds, t)
        sweep_rows.append({"threshold": round(t, 4), "accuracy": e["accuracy"], "FP": e["FP"], "FN": e["FN"],
                           "macro_f1": e["macro_f1"],
                           "pairs with early exit": _x_early_pairs(current_run, t, batch)})
    sweep_rows = sorted({r["threshold"]: r for r in sweep_rows}.values(), key=lambda r: r["threshold"])
    _x_write_csv(out_dir / "threshold_sweep_current.csv", sweep_rows)

    error_rows = []
    for fam, f in finals.items():
        run, t = f["run"], f["threshold"]
        for r, v, best, pruned in zip(records, run["values"], run["best"], run["pruned"]):
            pred = "S" if v >= t else "NS"
            if pred != r["gold"]:
                error_rows.append({"config": fam, "type": "FP" if pred == "S" else "FN", "index": r["index"],
                                   "fact": r["fact"], "label": r["label"], "score": round(v, 4),
                                   "fact word overlap": round(r["fact_overlap"][f["cfg"]["metric"]], 3),
                                   "pruned": pruned, "best evidence": best})
    _x_write_csv(out_dir / "errors_by_config.csv", error_rows)
    sets = {fam: {(e["index"], e["type"]) for e in error_rows if e["config"] == fam} for fam in finals}
    error_diff = [{"config": fam, "errors": len(sets[fam]), "fixed vs A": len(sets["A"] - sets[fam]),
                   "new vs A": len(sets[fam] - sets["A"])} for fam in finals]

    rec_fam = max(finals, key=lambda k: (round(finals[k]["nested"]["cv_mean"], 4), -finals[k]["pairs"]))
    rec = finals[rec_fam]
    rc = rec["cfg"]
    recommendation = [
        "    DEFAULT_THRESHOLD = %.4f" % rec["threshold"],
        "    FACT_PRUNE_THRESHOLD = %s" % rc["fact_tau"],
        "    PRUNE_METRIC = \"%s\"" % rc["metric"],
        "    EVIDENCE_PRUNE_THRESHOLD = %s" % rc["ev_tau"],
        "    EVIDENCE_KEEP_TOP_K = %d" % _x_EVID_RULES[rc["rule"]][0],
        "    EVIDENCE_KEEP_LEAD = %s" % _x_EVID_RULES[rc["rule"]][1]]

    timing_rows = None
    if args.timing:
        def kwargs(cfg, t, early):
            top_k, lead = _x_EVID_RULES[cfg["rule"]]
            return {"threshold": t, "fact_prune_threshold": cfg["fact_tau"], "evidence_prune_threshold": cfg["ev_tau"],
                    "prune_metric": cfg["metric"], "evidence_keep_top_k": top_k, "evidence_keep_lead": lead,
                    "evidence_mode": cfg["mode"], "titled_pronoun_units": cfg["titled"],
                    "min_evidence_words": cfg["min_words"], "early_exit": early}

        def expected(run, t):
            return ["S" if v >= t else "NS" for v in run["values"]]

        jobs = [{"name": "batch %d (sweep)" % b, "batch_size": b, "facts": args.sweep_facts,
                 "checker": kwargs(_x_BASE, _x_PREVIOUS["threshold"], False),
                 "expected": expected(previous_run, _x_PREVIOUS["threshold"])} for b in args.batch_sizes]
        jobs.append({"name": "previous submission (batch 1)", "batch_size": 1, "facts": args.timing_facts,
                     "checker": kwargs(_x_BASE, _x_PREVIOUS["threshold"], False),
                     "expected": expected(previous_run, _x_PREVIOUS["threshold"])})
        for fam, f in finals.items():
            jobs.append({"name": fam + ": " + f["title"], "batch_size": batch, "facts": args.timing_facts,
                         "checker": kwargs(f["cfg"], f["threshold"], False),
                         "expected": expected(f["run"], f["threshold"])})
        jobs.append({"name": rec_fam + " + early exit (recommended)", "batch_size": batch,
                     "facts": args.timing_facts, "checker": kwargs(rec["cfg"], rec["threshold"], True),
                     "expected": expected(rec["run"], rec["threshold"])})
        timing_rows = _x_run_timing(args, jobs)
        _x_write_csv(out_dir / "timing.csv", timing_rows)

    figures = _x_figures(Path(args.fig_dir), cost_rows, finals,
                         {fam: f["run"] for fam, f in finals.items()}, golds, timing_rows)

    # Pareto front: settings more accurate (grouped CV) than every cheaper setting.
    pareto, best_so_far = [], -1.0
    for r in sorted(cost_rows, key=lambda r: (r["pairs"], -r["cv_acc"])):
        if r["cv_acc"] > best_so_far:
            pareto.append(r)
            best_so_far = r["cv_acc"]

    kinds = Counter(u["kind"] for r in records for u in r["units"])
    L = ["# Part 2 experiment results", "",
         "Facts: %d (%d S, %d NS incl. IR), %d people (CV groups). Cached units: %s. Cache fingerprint `%s`." % (
             len(records), golds.count("S"), golds.count("NS"), len(set(groups)), dict(kinds), _x_fingerprint()),
         "", "All numbers are on the supplied development set (the only labelled data). **cv** = cross-validation "
         "grouped by person, %d repeats x 5 folds: the pruning setting and the decision threshold are re-tuned on "
         "the training people and the held-out people are predicted, so cv numbers are the honest estimate. **dev** "
         "= the final setting and threshold applied to all 221 facts (optimistic). **pairs** = premise/hypothesis "
         "pairs sent to DeBERTa for all 221 facts; with early exit, scoring stops as soon as one unit reaches the "
         "threshold (same predictions)." % R, "",
         "## A-D: pruning configurations", "",
         _x_md_table(main_rows, ["config", "final setting", "threshold", "cv_acc (nested)", "cv_acc_sd",
                                 "cv_acc (final setting fixed)", "cv S_precision",
                                 "cv S_recall", "cv S_f1", "cv NS_f1", "cv macro_f1",
                                 "cv CM [TP FN; FP TN] per repeat"]), "",
         _x_md_table(main_rows, ["config", "dev_acc", "dev S_precision", "dev S_recall", "dev S_f1",
                                 "dev NS_precision", "dev NS_recall", "dev NS_f1", "dev macro_f1",
                                 "dev CM [TP FN; FP TN]", "FP", "FN"]), "",
         _x_md_table(main_rows, ["config", "facts pruned", "S facts pruned", "pairs", "pairs_saved",
                                 "pairs with early exit", "early_saved", "model inputs (windows)",
                                 "premise tokens"]), "",
         "Previous submission (context, no pruning, t=0.436): dev %d/%d = %.2f%%, cv %.2f%%, %d pairs." % (
             previous["dev"]["correct"], len(records), 100 * previous["dev"]["accuracy"], 100 * previous["cv"],
             previous["pairs"]), "",
         "Errors of each final setting compared with A: " + "; ".join(
             "%s %d errors (%d fixed, %d new)" % (r["config"], r["errors"], r["fixed vs A"], r["new vs A"])
             for r in error_diff) + ". Lists: errors_by_config.csv.", ""]
    if timing_rows:
        L += ["## Measured time and memory (real model, fresh process per row)", "",
              _x_md_table(timing_rows, ["name", "facts", "batch_size", "seconds", "seconds_per_fact", "pairs",
                                        "model_inputs", "forward_passes", "peak_rss_mb", "inference_overhead_mb",
                                        "cuda_peak_mb", "accuracy", "FP", "FN", "matches_offline_analysis"]), "",
              "peak_rss_mb is the process high-water mark (model weights included); inference_overhead_mb is the "
              "part added by inference. matches_offline_analysis checks that the real predict() gives exactly the "
              "predictions of the offline analysis.", ""]
    L += ["## Accuracy versus cost", "",
          "Every setting searched (accuracy_vs_cost_all_settings.csv); best grouped-CV accuracy for a given budget:", "",
          _x_md_table(pareto, ["family", "setting", "cv_acc", "pairs", "pairs_saved"]), "",
          "## Evidence selection and decision strategy (no pruning)", "",
          _x_md_table(selection_rows, ["evidence units", "threshold", "dev_acc", "cv_acc", "FP", "FN", "pairs",
                                       "model inputs"]), "",
          "## Does pruning remove relevant evidence or supported facts?", "",
          "Fact level: supported facts pruned. Evidence level: facts predicted S without pruning whose best evidence "
          "unit is removed (their score drops). 'safe' keeps the 3 highest-overlap units and the article's lead "
          "sentence; 'fuzzy_recall' also matches 5-letter prefixes.", "",
          _x_md_table(safety_rows, ["level", "metric", "rule", "tau", "units kept", "facts pruned", "S facts pruned",
                                    "best evidence lost (S-predicted facts)", "of which gold S", "cv_acc"]), "",
          "## Are the current thresholds still optimal?", "",
          "Current class constants (fact tau %s %s, evidence tau %s, threshold %.4f): dev %d/%d = %.2f%%; grouped CV "
          "with these pruning settings (threshold re-tuned in the folds) %.2f%%; pairs %d, with early exit %d." % (
              C.FACT_PRUNE_THRESHOLD, C.PRUNE_METRIC, C.EVIDENCE_PRUNE_THRESHOLD, C.DEFAULT_THRESHOLD,
              current["dev"]["correct"], len(records), 100 * current["dev"]["accuracy"],
              100 * current["cv_retuned"], current["pairs"], current["early_pairs"]), "",
          _x_md_table(stability_rows, list(stability_rows[0].keys())), "",
          _x_md_table(sweep_rows, ["threshold", "accuracy", "FP", "FN", "macro_f1", "pairs with early exit"]), "",
          "## Recommended class constants (best nested grouped-CV accuracy, ties: fewer pairs): " + rec_fam, "",
          "```python"] + recommendation + ["```", ""]
    if figures:
        L += ["Figures: " + ", ".join("report/current_run_figures/" + f for f in figures), ""]
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\nRecommended (" + rec_fam + "):\n" + "\n".join(recommendation))
    print("Full results:", out_dir / "summary.md")
    return 0


# --- PART 1: COMPARISON COMMAND ---
_PART1_DOC = """
Part 1 experiments: compare overlap metrics, preprocessing and thresholds.
Report ROC-AUC, confidence intervals, development accuracy and
cross-validation results grouped by person. Save tables and optional figures.
"""
_P1_METRICS = ("recall", "precision", "f1", "jaccard", "tfidf_cosine", "idf_recall", "rouge2", "rougeL", "bleu")
_P1_LABELS = {"recall": "recall (= ROUGE-1 recall)", "precision": "precision", "f1": "F1", "jaccard": "Jaccard",
              "tfidf_cosine": "TF-IDF cosine", "idf_recall": "IDF-weighted recall", "rouge2": "ROUGE-2 recall",
              "rougeL": "ROUGE-L recall", "bleu": "BLEU-4"}


def _p1_auc(values, positive):
    """ ROC-AUC = Mann-Whitney U / (n_pos x n_neg) with average ranks for ties (a tie counts 1/2). """
    v = np.asarray(values, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2.0)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _p1_bootstrap(values, reference, positive, samples, seed):
    """ Paired bootstrap over facts: 95% CI of AUC(values) and of AUC(values) - AUC(reference). """
    rng = np.random.default_rng(seed)
    v, ref, pos = np.asarray(values, float), np.asarray(reference, float), np.asarray(positive, bool)
    aucs, diffs = [], []
    for _ in range(samples):
        idx = rng.integers(0, len(v), len(v))
        if pos[idx].all() or not pos[idx].any():
            continue
        a = _p1_auc(v[idx], pos[idx])
        aucs.append(a)
        diffs.append(a - _p1_auc(ref[idx], pos[idx]))
    q = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    return q(aucs), q(diffs)


def _p1_roc(values, positive):
    points = [(0.0, 0.0)]
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    for t in sorted(set(values), reverse=True):
        tp = sum(1 for v, p in zip(values, positive) if p and v >= t)
        fp = sum(1 for v, p in zip(values, positive) if not p and v >= t)
        points.append((fp / n_neg, tp / n_pos))
    return points


def _part1_main(argv):
    parser = argparse.ArgumentParser(prog="factcheck_experiments.py part1", description=_PART1_DOC,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--out_dir", default="results/current_run/part1")
    parser.add_argument("--fig_dir", default="report/current_run_figures")
    parser.add_argument("--cv_repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    os.chdir(_x_ROOT)
    examples = _x_load_examples(args.labels_path, args.passages_path)
    golds = [_x_gold_of(e) for e in examples]
    positive = [g == "S" for g in golds]
    groups = [e.passages[0]["title"] if e.passages else "" for e in examples]
    print("Read", len(examples), "examples (%d S, %d NS)" % (golds.count("S"), golds.count("NS")))
    R = args.cv_repeats
    chosen = WordRecallThresholdFactChecker
    preprocessors = {}

    def scores(metric="recall", aggregation=None, drop_title=None, **pre_changes):
        settings = dict(chosen.PREPROCESSING, **pre_changes)
        key = tuple(sorted(settings.items()))
        if key not in preprocessors:
            preprocessors[key] = TextPreprocessor(**settings)
        pre = preprocessors[key]
        aggregation = chosen.AGGREGATION if aggregation is None else aggregation
        drop_title = chosen.DROP_TITLE if drop_title is None else drop_title
        started = time.perf_counter()
        values = [word_overlap_score(e.fact, e.passages, pre, metric, aggregation, drop_title) for e in examples]
        return values, time.perf_counter() - started

    reference, _ = scores()

    def row(name, values, seconds=None, **extra):
        t, acc = _x_select_threshold(values, golds)
        (lo, hi), (dlo, dhi) = _p1_bootstrap(values, reference, positive, args.bootstrap, args.seed)
        auc = _p1_auc(values, positive)
        cv = _x_grouped_cv({"x": {"values": values, "pairs": 0}}, golds, groups, R)
        mean_s = statistics.mean(v for v, p in zip(values, positive) if p)
        mean_ns = statistics.mean(v for v, p in zip(values, positive) if not p)
        r = dict(extra, configuration=name, auc=round(auc, 4), auc_ci="[%.3f, %.3f]" % (lo, hi),
                 delta_auc=round(auc - _p1_auc(reference, positive), 4), delta_ci="[%.3f, %.3f]" % (dlo, dhi),
                 significance=("worse" if dhi < 0 else "better" if dlo > 0 else "not significant")
                 if name != "chosen" else "reference",
                 threshold=round(t, 4), dev_acc=acc, cv_acc=cv["cv_mean"], cv_acc_sd=cv["cv_sd"],
                 mean_score_S=round(mean_s, 3), mean_score_NS=round(mean_ns, 3))
        if seconds is not None:
            r["seconds"] = round(seconds, 3)
        return r

    # 1. metric comparison
    print("Comparing metrics ...", flush=True)
    metric_rows, roc_curves = [], {}
    for metric in _P1_METRICS:
        for aggregation in AGGREGATIONS:
            values, seconds = scores(metric, aggregation)
            name = "chosen" if (metric, aggregation) == (chosen.METRIC, chosen.AGGREGATION) else \
                "%s, %s" % (_P1_LABELS[metric], aggregation)
            metric_rows.append(row(name, values, seconds, metric=_P1_LABELS[metric], aggregation=aggregation,
                                   stop_words="removed"))
            if aggregation == "max":
                roc_curves[_P1_LABELS[metric]] = _p1_roc(values, positive)
    for metric in ("rouge2", "rougeL", "bleu"):
        values, seconds = scores(metric, "max", remove_stopwords=False)
        metric_rows.append(row("%s, max, stop words kept" % _P1_LABELS[metric], values, seconds,
                               metric=_P1_LABELS[metric], aggregation="max", stop_words="kept"))
    best_per_metric = []
    for metric in _P1_METRICS:
        candidates = [r for r in metric_rows if r["metric"] == _P1_LABELS[metric]]
        best_per_metric.append(max(candidates, key=lambda r: r["auc"]))
    best_per_metric.sort(key=lambda r: -r["auc"])

    # 2. preprocessing ablation (one factor at a time)
    print("Preprocessing ablation ...", flush=True)
    ablation = [row("chosen", reference, scores()[1], change="none (chosen configuration)")]
    variants = [("stemming: none", {"normalizer": "none"}), ("stemming: Snowball", {"normalizer": "snowball"}),
                ("stemming: spaCy lemmatiser", {"normalizer": "lemma"}), ("stop words kept", {"remove_stopwords": False}),
                ("person's name kept", {"drop_title": False}), ("case kept", {"lowercase": False}),
                ("min token length 1", {"min_token_len": 1}), ("aggregation: concatenated passages",
                                                                 {"aggregation": "concat"}),
                ("aggregation: best sentence", {"aggregation": "sentence_max"})]
    for change, kwargs in variants:
        try:
            values, seconds = scores("recall", **kwargs)
        except (OSError, IOError) as error:  # en_core_web_sm not installed
            print("  skipped", change + ":", str(error).splitlines()[0][:100])
            continue
        ablation.append(row(change, values, seconds, change=change))

    # 3. the chosen classifier, exactly as the driver runs it
    print("Evaluating the chosen classifier ...", flush=True)
    checker = chosen()
    predictions = [checker.predict(e.fact, e.passages) for e in examples]
    cm = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for p, g in zip(predictions, golds):
        cm[("T" if p == g else "F") + ("P" if p == "S" else "N")] += 1
    final = _x_metrics(cm)
    final_cv = _x_grouped_cv({"x": {"values": reference, "pairs": 0}}, golds, groups, R)
    sweep = []
    for t in sorted(set([round(x / 20, 2) for x in range(2, 20)] + [checker.threshold])):
        e = _x_evaluate(reference, golds, t)
        sweep.append({"threshold": round(t, 4), "accuracy": e["accuracy"], "FP": e["FP"], "FN": e["FN"],
                      "S_f1": e["S_f1"], "NS_f1": e["NS_f1"]})
    bins = []
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        inside = [g for v, g in zip(reference, golds) if lo <= v < hi or (k == 9 and v == 1.0)]
        bins.append({"score bin": "[%.1f, %.1f%s" % (lo, hi, "]" if k == 9 else ")"), "S": inside.count("S"),
                     "NS": inside.count("NS")})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _x_write_csv(out_dir / "metric_comparison.csv", metric_rows)
    _x_write_csv(out_dir / "preprocessing_ablation.csv", ablation)
    _x_write_csv(out_dir / "threshold_sweep.csv", sweep)
    _x_write_csv(out_dir / "score_distribution.csv", bins)

    figures = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.6, 5))
        for label, points in roc_curves.items():
            ax.plot([p[0] for p in points], [p[1] for p in points], label=label, linewidth=1.2)
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "part1_roc_metrics.png", dpi=200)
        plt.close(fig)
        figures.append("part1_roc_metrics.png")
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ordered = best_per_metric[::-1]  # best at the top
        names = ["%s (%s)" % (r["metric"], r["aggregation"]) for r in ordered]
        aucs = [r["auc"] for r in ordered]
        errors = [[r["auc"] - float(r["auc_ci"][1:-1].split(",")[0]) for r in ordered],
                  [float(r["auc_ci"][1:-1].split(",")[1]) - r["auc"] for r in ordered]]
        ax.barh(range(len(names)), aucs, xerr=errors, capsize=3, color="#1f77b4")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("ROC-AUC with 95% bootstrap CI (best aggregation of each metric)")
        ax.set_xlim(0.5, 1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / "part1_auc_by_metric.png", dpi=200)
        plt.close(fig)
        figures.append("part1_auc_by_metric.png")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot([r["threshold"] for r in sweep], [100 * r["accuracy"] for r in sweep], marker="o")
        ax.axvline(checker.threshold, color="grey", linestyle=":")
        ax.axhline(75, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("threshold on word-overlap recall")
        ax.set_ylabel("dev accuracy (%)")
        fig.tight_layout()
        fig.savefig(fig_dir / "part1_threshold_accuracy.png", dpi=200)
        plt.close(fig)
        figures.append("part1_threshold_accuracy.png")
    except ImportError:
        print("matplotlib not installed: figures skipped (optional).")

    cols = ["configuration", "auc", "auc_ci", "delta_auc", "delta_ci", "significance", "threshold", "dev_acc",
            "cv_acc", "cv_acc_sd", "mean_score_S", "mean_score_NS"]
    L = ["# Part 1 experiment results", "",
         "Facts: %d (%d S, %d NS incl. IR), %d people. AUC = ROC-AUC (threshold-free); CI = paired bootstrap 95%% "
         "interval over facts (%d resamples); delta = AUC minus the chosen configuration (recall, max over passages, "
         "chosen preprocessing). dev_acc = accuracy at the best threshold on all facts (optimistic); cv_acc = "
         "cross-validation grouped by person (%d repeats x 5 folds, threshold re-tuned on the training people)." % (
             len(examples), golds.count("S"), golds.count("NS"), len(set(groups)), args.bootstrap, R), "",
         "## Final classifier (what factchecking_main.py --mode word_overlap runs)", "",
         "Threshold %.4f. Accuracy %d/%d = %.2f%%. Confusion matrix [S: TP FN; NS: FP TN] = [%d %d; %d %d]. "
         "S precision/recall/F1 %.3f/%.3f/%.3f; NS %.3f/%.3f/%.3f. Grouped CV accuracy %.2f%% +- %.2f." % (
             checker.threshold, final["correct"], final["total"], 100 * final["accuracy"], final["TP"], final["FN"],
             final["FP"], final["TN"], final["S_precision"], final["S_recall"], final["S_f1"], final["NS_precision"],
             final["NS_recall"], final["NS_f1"], 100 * final_cv["cv_mean"], 100 * final_cv["cv_sd"]), "",
         "## Best aggregation of each metric", "", _x_md_table(best_per_metric, ["metric", "aggregation"] + cols[1:]),
         "", "## All metric / aggregation combinations", "",
         _x_md_table(metric_rows, ["metric", "aggregation", "stop_words"] + cols[1:] + ["seconds"]), "",
         "## Preprocessing ablation (recall; one setting changed at a time)", "",
         _x_md_table(ablation, ["change"] + cols[1:] + ["seconds"]), "",
         "## Threshold sweep of the final classifier", "",
         _x_md_table(sweep, ["threshold", "accuracy", "FP", "FN", "S_f1", "NS_f1"]), "",
         "## Score distribution of the final classifier", "", _x_md_table(bins, ["score bin", "S", "NS"]), ""]
    if figures:
        L += ["Figures: " + ", ".join("report/current_run_figures/" + f for f in figures), ""]
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:14]))
    print("Full results:", out_dir / "summary.md")
    return 0


# --- PART 3: ERROR-ANALYSIS COMMAND ---
_ERRORS_DOC = """
Create the Part 3 error worksheet for the current entailment classifier.
Reuse cached scores and list errors with evidence and pruning details.
Select up to 10 errors of each type using a fixed random seed.
Save CSV and Markdown worksheets with category and notes fields to fill in.
"""


def _error_worksheet(argv):
    parser = argparse.ArgumentParser(prog="factcheck_experiments.py errors", description=_ERRORS_DOC,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--cache", default="results/current_run/part2/unit_scores.jsonl")
    parser.add_argument("--out_dir", default="results/current_run/part3")
    parser.add_argument("--per_type", type=int, default=10, help="errors of each type selected for analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max_facts", type=int, default=None, help=argparse.SUPPRESS)  # for the self-test
    args = parser.parse_args(argv)
    os.chdir(_x_ROOT)
    examples = _x_load_examples(args.labels_path, args.passages_path)
    examples = examples[:args.max_facts] if args.max_facts else examples
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    records = _x_score_units(examples, cache, args.cuda, None)
    _x_annotate(records, examples)
    cfg, threshold = _x_current_cfg(), EntailmentFactChecker.DEFAULT_THRESHOLD
    run = _x_run(records, cfg)
    golds = [r["gold"] for r in records]
    predictions = ["S" if v >= threshold else "NS" for v in run["values"]]
    cm = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for p, g in zip(predictions, golds):
        cm[("T" if p == g else "F") + ("P" if p == "S" else "N")] += 1
    final = _x_metrics(cm)
    print("Final classifier on the cached scores: %d/%d = %.2f%%, [TP FN; FP TN] = [%d %d; %d %d]" % (
        final["correct"], final["total"], 100 * final["accuracy"], cm["TP"], cm["FN"], cm["FP"], cm["TN"]))

    kinds = _x_kinds(cfg)
    fmt = lambda p: "E %.3f / N %.3f / C %.3f" % tuple(p)
    errors = {"FP": [], "FN": []}
    for i, (r, pred, gold) in enumerate(zip(records, predictions, golds)):
        if pred == gold:
            continue
        kind = "FP" if pred == "S" else "FN"
        units = [u for u in r["units"] if u["kind"] in kinds]
        top = sorted(units, key=lambda u: -u["p"][0])[:3]
        contra = max(units, key=lambda u: u["p"][2]) if units else None
        lead = next((u["text"] for u in r["units"] if u["kind"] == "sentence" and u["lead"]), "")
        row = {"id": "", "type": kind, "selected": False, "fact": r["fact"], "gold_label": r["label"],
               "prediction": pred, "score": round(run["values"][i], 4),
               "word_overlap": round(r["fact_overlap"][cfg["metric"]], 3), "pruned": run["pruned"][i],
               "scored_in_final_run": not run["pruned"][i]}
        for k in range(3):
            row["evidence_%d" % (k + 1)] = top[k]["text"] if k < len(top) else ""
            row["probs_%d" % (k + 1)] = fmt(top[k]["p"]) if k < len(top) else ""
        row["most_contradicting"] = contra["text"] if contra else ""
        row["most_contradicting_probs"] = fmt(contra["p"]) if contra else ""
        row["lead_sentence"] = lead
        row["category"] = ""
        row["notes"] = ""
        errors[kind].append(row)
    rng = random.Random(args.seed)
    for kind in ("FP", "FN"):
        rows = errors[kind]
        chosen = set(rng.sample(range(len(rows)), min(args.per_type, len(rows))))
        for n, row in enumerate(rows):
            row["id"] = "%s%02d" % (kind, n + 1)
            row["selected"] = n in chosen
    rows = errors["FP"] + errors["FN"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _x_write_csv(out_dir / "error_worksheet.csv", rows)

    L = ["# Error-analysis worksheet (Part 3)", "",
         "Final classifier: %s, threshold %.4f. Dev accuracy %d/%d = %.2f%%; %d false positives, %d false "
         "negatives. Seeded random selection (seed %d): %d FP and %d FN marked SELECTED." % (
             _x_cfg_name(cfg), threshold, final["correct"], final["total"], 100 * final["accuracy"], cm["FP"],
             cm["FN"], args.seed, sum(r["selected"] for r in errors["FP"]), sum(r["selected"] for r in errors["FN"])),
         "", "Probabilities are E(ntailment) / N(eutral) / C(ontradiction). A pruned fact never reached DeBERTa in the "
         "final run; its evidence is shown to judge whether pruning caused the error.", ""]
    for kind, title in (("FP", "False positives (predicted S, gold NS or IR)"),
                        ("FN", "False negatives (predicted NS, gold S)")):
        L += ["## " + title, ""]
        for row in errors[kind]:
            L += ["### %s%s: %s" % (row["id"], " (SELECTED)" if row["selected"] else "", row["fact"]), "",
                  "- gold %s, predicted %s, score %.4f, word overlap %.3f%s" % (
                      row["gold_label"], row["prediction"], row["score"], row["word_overlap"],
                      ", PRUNED before DeBERTa" if row["pruned"] else "")]
            for k in range(1, 4):
                if row["evidence_%d" % k]:
                    L.append("- evidence %d (%s): %s" % (k, row["probs_%d" % k], row["evidence_%d" % k]))
            if row["most_contradicting"]:
                L.append("- most contradicting (%s): %s" % (row["most_contradicting_probs"], row["most_contradicting"]))
            if row["lead_sentence"]:
                L.append("- article lead sentence: " + row["lead_sentence"])
            L += ["- category: ", "- notes: ", ""]
    L += ["## Summary table to fill in", "", "| category | definition (1-2 sentences) | FP | FN | total |",
          "|---|---|---|---|---|", "| | | | | |", "| | | | | |", "| | | | | |", ""]
    (out_dir / "error_worksheet.md").write_text("\n".join(L), encoding="utf-8")
    print("%d false positives and %d false negatives written to %s and %s" % (
        len(errors["FP"]), len(errors["FN"]), out_dir / "error_worksheet.csv", out_dir / "error_worksheet.md"))
    return 0


# --- PRUNED AND UNPRUNED COMPARISON ---
_COMPARE_DOC = """
Compare entailment with and without pruning, loading DeBERTa once.
Use current settings for pruning and threshold 0.436 without pruning.
Override the latter with --no_pruning_threshold.
Save the comparison and changed predictions without modifying factcheck.py.
"""


def _compare_pruning(argv):
    parser = argparse.ArgumentParser(prog="factcheck_experiments.py compare", description=_COMPARE_DOC,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no_pruning_threshold", type=float, default=0.436032310128212,
                        help="decision threshold of the run without pruning")
    parser.add_argument("--out_dir", default="results/current_run/part2")
    parser.add_argument("--max_facts", type=int, default=None, help=argparse.SUPPRESS)  # for the self-test
    args = parser.parse_args(argv)
    os.chdir(_x_ROOT)
    import factchecking_main as fm
    from tqdm import tqdm

    examples = fm.read_fact_examples(args.labels_path, fm.read_passages(args.passages_path))
    examples = examples[:args.max_facts] if args.max_facts else examples
    print("Read " + repr(len(examples)) + " examples")
    model, tokenizer = _x_load_model(args.cuda)
    C = EntailmentFactChecker
    runs = [("WITH pruning (factcheck.py defaults)", {},
             "fact %s >= %s, evidence pruning %s, threshold %.4f" % (
                 C.PRUNE_METRIC, C.FACT_PRUNE_THRESHOLD,
                 "off" if C.EVIDENCE_PRUNE_THRESHOLD == 0 else ">= %s" % C.EVIDENCE_PRUNE_THRESHOLD,
                 C.DEFAULT_THRESHOLD)),
            ("WITHOUT pruning", {"fact_prune_threshold": 0.0, "evidence_prune_threshold": 0.0,
                                 "threshold": args.no_pruning_threshold},
             "no fact or evidence pruning, threshold %.4f" % args.no_pruning_threshold)]
    labels = ["S", "NS"]
    results = []
    for title, kwargs, settings in runs:
        print("\n" + "=" * 100 + "\n" + title + ": " + settings + "\n" + "=" * 100)
        ent_model = EntailmentModel(model, tokenizer, args.cuda)
        checker = EntailmentFactChecker(ent_model, **kwargs)
        confusion = [[0, 0], [0, 0]]
        predictions = []
        started = time.perf_counter()
        for example in tqdm(examples):
            gold = labels.index("NS" if example.label == "IR" else example.label)
            prediction = checker.predict(example.fact, example.passages)
            confusion[gold][labels.index(prediction)] += 1
            predictions.append(prediction)
        seconds = time.perf_counter() - started
        fm.print_eval_stats(confusion, labels)  # the driver's own output format
        metrics = _x_metrics({"TP": confusion[0][0], "FN": confusion[0][1], "FP": confusion[1][0],
                              "TN": confusion[1][1]})
        results.append({"run": title, "settings": settings, "metrics": metrics, "seconds": seconds,
                        "pairs": ent_model.premises_scored, "forward_passes": ent_model.forward_passes,
                        "pruned": checker.stats["pruned_facts"], "predictions": predictions})
        print("time %.1f s (%.2f s/fact), %d premise/hypothesis pairs, %d forward passes, %d facts pruned" % (
            seconds, seconds / len(examples), ent_model.premises_scored, ent_model.forward_passes,
            checker.stats["pruned_facts"]))

    # side-by-side report
    rows = [("Accuracy", lambda r: "%d/%d = %.2f%%" % (r["metrics"]["correct"], r["metrics"]["total"],
                                                       100 * r["metrics"]["accuracy"])),
            ("Confusion matrix [S: TP FN; NS: FP TN]", lambda r: "[%d %d; %d %d]" % (
                r["metrics"]["TP"], r["metrics"]["FN"], r["metrics"]["FP"], r["metrics"]["TN"]))]
    for label in labels:
        for name in ("precision", "recall", "f1"):
            rows.append(("%s %s" % (label, name), lambda r, k=label + "_" + name: "%.2f%%" % (100 * r["metrics"][k])))
    rows += [("Macro F1", lambda r: "%.2f%%" % (100 * r["metrics"]["macro_f1"])),
             ("False positives / false negatives", lambda r: "%d / %d" % (r["metrics"]["FP"], r["metrics"]["FN"])),
             ("Facts pruned (no model call)", lambda r: str(r["pruned"])),
             ("Premise/hypothesis pairs scored", lambda r: str(r["pairs"])),
             ("Model forward passes", lambda r: str(r["forward_passes"])),
             ("Time", lambda r: "%.1f s (%.2f s/fact)" % (r["seconds"], r["seconds"] / len(examples)))]
    with_p, without_p = results
    table = ["| | " + with_p["run"] + " | " + without_p["run"] + " |", "|---|---|---|"]
    table += ["| %s | %s | %s |" % (name, fn(with_p), fn(without_p)) for name, fn in rows]
    changes = []
    for e, a, b in zip(examples, with_p["predictions"], without_p["predictions"]):
        if a != b:
            gold = "NS" if e.label == "IR" else e.label
            changes.append({"fact": e.fact, "label": e.label, "with pruning": a, "without pruning": b,
                            "better with": "pruning" if a == gold else "no pruning"})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gained = sum(1 for c in changes if c["better with"] == "pruning")
    lines = ["# Entailment with and without pruning", "",
             "WITH: " + with_p["settings"] + ". WITHOUT: " + without_p["settings"] + ".", "",
             "\n".join(table), "",
             "%d facts get a different prediction: %d are correct only with pruning, %d only without. "
             "Pairs saved by pruning: %.1f%%." % (len(changes), gained, len(changes) - gained,
                                                  100 * (1 - with_p["pairs"] / max(1, without_p["pairs"]))), ""]
    (out_dir / "compare_pruning.md").write_text("\n".join(lines), encoding="utf-8")
    _x_write_csv(out_dir / "compare_pruning_changes.csv", changes)
    print("\n" + "\n".join(lines))
    print("Saved:", out_dir / "compare_pruning.md", "and", out_dir / "compare_pruning_changes.csv")
    return 0


# --- SELF-TESTS ---
def _tiny_tokenizer(max_length=48):
    import string
    import tempfile
    from transformers import BertTokenizerFast
    """ Character-level WordPiece vocabulary: every word becomes several tokens, so windows are easy to trigger. """
    symbols = list(string.ascii_lowercase + string.digits)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + symbols + ["##" + s for s in symbols] + \
        list(".,;:!?\"'()-")
    directory = tempfile.mkdtemp()
    with open(os.path.join(directory, "vocab.txt"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(vocab))
    tokenizer = BertTokenizerFast(os.path.join(directory, "vocab.txt"))
    tokenizer.model_max_length = max_length
    return tokenizer


def _tiny_model(tokenizer, id2label=None, seed=0):
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
    torch.manual_seed(seed)
    id2label = id2label or {0: "entailment", 1: "neutral", 2: "contradiction"}
    config = DebertaV2Config(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=64, max_position_embeddings=512,
                             relative_attention=True, position_buckets=64, norm_rel_ebd="layer_norm",
                             share_att_key=True, pos_att_type=["p2c", "c2p"], position_biased_input=False,
                             type_vocab_size=0, num_labels=3, id2label=id2label,
                             label2id={v: k for k, v in id2label.items()}, pad_token_id=0)
    return DebertaV2ForSequenceClassification(config).eval()


def _self_test(argv):
    import tempfile
    import unittest
    from types import SimpleNamespace
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(ROOT, "data", "dev_labeled_ChatGPT.jsonl")

    _T_PREMISES = ["Alice was born in 1950.", "She sang.",
                "Bob Smith played football for many years in several clubs across Europe and later coached young players "
                "in his home town, where he also opened a small shop that sold sports equipment to local schools.",
                "", "Short one."]


    class _FakeEntailment(object):
        """ Records the calls; P(entailment) = 0.9 if the premise contains the given keyword. """
        def __init__(self, keyword="sang"):
            self.keyword, self.calls, self.premises = keyword, 0, []

        def score_batch(self, premises, hypothesis):
            self.calls += 1
            self.premises.extend(premises)
            return [0.9 if self.keyword in p else 0.1 for p in premises]


    class EntailmentModelTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.tokenizer = _tiny_tokenizer()
            cls.model = _tiny_model(cls.tokenizer)

        def test_batched_equals_one_pair_at_a_time(self):
            single = EntailmentModel(self.model, self.tokenizer, batch_size=1)
            batched = EntailmentModel(self.model, self.tokenizer, batch_size=8)
            tight = EntailmentModel(self.model, self.tokenizer, batch_size=8, max_batch_tokens=60)
            reference = [single.predict_probs([p], "Alice sang.")[0] for p in _T_PREMISES]
            for ent_model in (single, batched, tight):
                for got, want in zip(ent_model.predict_probs(_T_PREMISES, "Alice sang."), reference):
                    for a, b in zip(got, want):
                        self.assertAlmostEqual(a, b, places=5)
            self.assertLess(batched.forward_passes, len(_T_PREMISES))

        def test_probabilities_are_valid_and_empty_premise_is_neutral(self):
            ent_model = EntailmentModel(self.model, self.tokenizer)
            probs = ent_model.predict_probs(_T_PREMISES, "Alice sang.")
            self.assertEqual(len(probs), len(_T_PREMISES))
            for p in probs:
                self.assertAlmostEqual(sum(p), 1.0, places=5)
            self.assertEqual(probs[3], [0.0, 1.0, 0.0])
            self.assertEqual(ent_model.check_entailment(_T_PREMISES[0], "Alice sang."), probs[0][0])

        def test_long_premise_uses_windows_and_takes_the_max(self):
            ent_model = EntailmentModel(self.model, self.tokenizer)
            hyp = self.tokenizer.encode("Alice sang.", add_special_tokens=False)
            ids = self.tokenizer.encode(_T_PREMISES[2], add_special_tokens=False, verbose=False)
            windows = ent_model._windows(ids, hyp)
            self.assertGreater(len(windows), 1)
            self.assertEqual(sum(len(w) for w in windows) >= len(ids), True)  # nothing dropped
            self.assertEqual(windows[-1][-1], ids[-1])                           # the end of the premise is covered
            scores = []
            for window in windows:  # every window scored on its own, without the class under test
                enc = self.tokenizer.prepare_for_model(window, pair_ids=hyp, return_tensors="pt", prepend_batch_axis=True,
                                                       verbose=False)
                self.assertLessEqual(enc["input_ids"].shape[1], ent_model.max_length)
                with torch.no_grad():
                    scores.append(torch.softmax(self.model(**enc).logits, -1)[0, 0].item())
            self.assertAlmostEqual(ent_model.check_entailment(_T_PREMISES[2], "Alice sang."), max(scores), places=5)

        def test_batches_respect_both_limits(self):
            ent_model = EntailmentModel(self.model, self.tokenizer, batch_size=3, max_batch_tokens=50)
            features = [(i, [1] * n, [0] * n) for i, n in enumerate([5, 40, 12, 12, 30, 7, 7, 7, 7])]
            seen = []
            for batch in ent_model._batches(features):
                self.assertLessEqual(len(batch), 3)
                width = max(len(features[i][1]) for i in batch)
                self.assertTrue(width * len(batch) <= 50 or len(batch) == 1)
                seen.extend(batch)
            self.assertEqual(sorted(seen), list(range(len(features))))

        def test_label_order_is_read_from_config(self):
            permuted = _tiny_model(self.tokenizer, {0: "contradiction", 1: "entailment", 2: "neutral"}, seed=1)
            ent_model = EntailmentModel(permuted, self.tokenizer)
            self.assertEqual((ent_model.entailment_index, ent_model.contradiction_index), (1, 0))
            enc = self.tokenizer("Alice was born in 1950.", "Alice sang.", return_tensors="pt")
            with torch.no_grad():
                raw = torch.softmax(permuted(**enc).logits, -1)[0].tolist()
            got = ent_model.predict_probs(["Alice was born in 1950."], "Alice sang.")[0]
            self.assertAlmostEqual(got[0], raw[1], places=5)  # entailment
            self.assertAlmostEqual(got[1], raw[2], places=5)  # neutral
            self.assertAlmostEqual(got[2], raw[0], places=5)  # contradiction

        def test_generic_labels_are_rejected_for_unknown_models(self):
            generic = _tiny_model(self.tokenizer, {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
            with self.assertRaises(ValueError):
                EntailmentModel(generic, self.tokenizer)


    class EntailmentFactCheckerTests(unittest.TestCase):
        PASSAGES = [{"title": "Alice Smith", "text": "<s>Alice Smith was a singer. She sang in Paris.</s>"
                                                     "<s>Personal life. She married Bob in 1970.</s>"}]

        def test_driver_constructor_uses_class_defaults(self):
            checker = EntailmentFactChecker(_FakeEntailment())
            self.assertEqual(checker.threshold, EntailmentFactChecker.DEFAULT_THRESHOLD)
            self.assertEqual(checker.fact_prune_threshold, EntailmentFactChecker.FACT_PRUNE_THRESHOLD)
            self.assertEqual(checker.evidence_mode, EntailmentFactChecker.EVIDENCE_MODE)

        def test_invalid_settings_are_rejected(self):
            for kwargs in ({"threshold": 0.0}, {"threshold": float("nan")}, {"fact_prune_threshold": 1.5},
                           {"evidence_prune_threshold": -0.1}, {"evidence_mode": "paragraph"},
                           {"prune_metric": "jaccard"}, {"evidence_keep_top_k": -1}):
                with self.assertRaises(ValueError):
                    EntailmentFactChecker(_FakeEntailment(), **kwargs)

        def test_fact_pruning_skips_the_model(self):
            fake = _FakeEntailment()
            checker = EntailmentFactChecker(fake, fact_prune_threshold=0.3)
            self.assertEqual(checker.predict("Alice Smith won an Olympic medal.", self.PASSAGES), "NS")
            self.assertEqual(fake.calls, 0)
            self.assertEqual(checker.stats["pruned_facts"], 1)
            self.assertEqual(checker.predict("Alice Smith sang in Paris.", self.PASSAGES), "S")
            self.assertEqual(fake.calls, 1)

        def test_no_pruning_runs_the_model(self):
            fake = _FakeEntailment()
            checker = EntailmentFactChecker(fake, fact_prune_threshold=0.0)
            checker.predict("Alice Smith won an Olympic medal.", self.PASSAGES)
            self.assertEqual(fake.calls, 1)

        def test_score_is_the_max_over_all_evidence(self):
            checker = EntailmentFactChecker(_FakeEntailment(), fact_prune_threshold=0.0)
            score, evidence = checker.score_details("Alice sang.", self.PASSAGES)
            self.assertEqual(score, 0.9)
            self.assertIn("sang", evidence)

        def test_evidence_modes(self):
            checker = EntailmentFactChecker(_FakeEntailment())
            units = checker.evidence_units(self.PASSAGES, "context")
            self.assertIn(("She sang in Paris.", "sentence"), units)
            self.assertIn(("Alice Smith was a singer. She sang in Paris.", "context"), units)
            self.assertNotIn("She sang in Paris. Personal life.", [t for t, _ in units])  # never across <s> segments
            self.assertTrue(all(k == "sentence" for _, k in checker.evidence_units(self.PASSAGES, "sentence")))
            self.assertEqual(len(checker.evidence_units(self.PASSAGES, "passage")), 1)

        def test_fragment_filter_and_evidence_pruning(self):
            fake = _FakeEntailment()
            checker = EntailmentFactChecker(fake, fact_prune_threshold=0.0, min_evidence_words=3)
            self.assertNotIn("Personal life.", checker.select_evidence("Alice married Bob.", self.PASSAGES))
            lexical = EntailmentFactChecker(fake, fact_prune_threshold=0.0, evidence_prune_threshold=0.5,
                                            evidence_keep_top_k=0, evidence_keep_lead=False)
            selected = lexical.select_evidence("Alice married Bob in 1970.", self.PASSAGES)
            self.assertTrue(selected)
            self.assertTrue(all("1970" in s for s in selected))
            safe = EntailmentFactChecker(fake, fact_prune_threshold=0.0, evidence_prune_threshold=0.5,
                                         evidence_keep_top_k=1, evidence_keep_lead=True)
            kept = safe.select_evidence("Alice married Bob in 1970.", self.PASSAGES)
            self.assertIn("Alice Smith was a singer.", kept)                   # lead sentence always kept
            self.assertEqual(kept[0], "She married Bob in 1970.")  # highest overlap first (ties: document order)

        def test_plan_evidence_keeps_top_k_and_orders_by_overlap(self):
            records = [{"text": "a b c", "lead": False}, {"text": "d e f", "lead": True},
                       {"text": "g h i", "lead": False}, {"text": "j", "lead": False}]
            overlaps = [0.1, 0.0, 0.6, 0.3]
            plan = EntailmentFactChecker.plan_evidence
            self.assertEqual(plan(records, overlaps, 0, 0.0, 3, True), [2, 3, 0, 1])   # no pruning: all, sorted
            self.assertEqual(plan(records, overlaps, 0, 0.5, 0, False), [2])
            self.assertEqual(plan(records, overlaps, 0, 0.5, 2, False), [2, 3])
            self.assertEqual(plan(records, overlaps, 0, 0.5, 0, True), [2, 1])
            self.assertEqual(plan(records, overlaps, 2, 0.0, 0, False), [2, 0, 1])    # "j" has 1 word

        def test_early_exit_gives_identical_predictions_with_fewer_pairs(self):
            import random as _random

            class Hashed(object):
                batch_size = 2

                def __init__(self):
                    self.pairs = 0

                def score_batch(self, premises, hypothesis):
                    self.pairs += len(premises)
                    return [_random.Random(p + hypothesis).random() for p in premises]

            passages = [{"title": "Alice Smith", "text": "<s>Alice Smith was a singer. She sang in Paris. She won a "
                                                         "prize in 1999. Her album sold well.</s><s>Personal life. "
                                                         "She married Bob in 1970. They moved to Rome.</s>"}]
            facts = ["Alice sang in Paris.", "Alice married Bob.", "Alice won a prize.", "Alice lived in Rome.",
                     "Alice was a painter.", "Alice released an album.", "Alice was born in 1950."]
            full, early = Hashed(), Hashed()
            for t in (0.3, 0.6, 0.9, 0.99):
                for fact in facts:
                    a = EntailmentFactChecker(full, threshold=t, fact_prune_threshold=0.0, early_exit=False)
                    b = EntailmentFactChecker(early, threshold=t, fact_prune_threshold=0.0, early_exit=True)
                    self.assertEqual(a.predict(fact, passages), b.predict(fact, passages))
            self.assertLess(early.pairs, full.pairs)

        def test_lead_sentence_and_pronoun_resolved_units(self):
            checker = EntailmentFactChecker(_FakeEntailment())
            records = checker.evidence_records(self.PASSAGES, "context", titled=True)
            lead = [r["text"] for r in records if r["lead"]]
            self.assertIn("Alice Smith was a singer.", lead)
            self.assertNotIn("Personal life.", lead)                           # 2nd segment is not the article lead
            titled = [r["text"] for r in records if r["kind"] == "titled"]
            self.assertIn("Alice Smith sang in Paris.", titled)
            self.assertIn("Alice Smith married Bob in 1970.", titled)
            other = [{"title": "Bob Jones (singer)", "text": "<s>In 1990 he left. His album sold well. They met.</s>"}]
            records = checker.evidence_records(other, "sentence", titled=True)
            self.assertFalse(any(r["lead"] for r in records))                  # chunk does not open the article
            titled = [r["text"] for r in records if r["kind"] == "titled"]
            self.assertEqual(titled, ["Bob Jones's album sold well."])         # possessive; "They" left alone
            self.assertEqual([r["kind"] for r in checker.evidence_records(self.PASSAGES, "context")].count("titled"),
                             0)                                                # off by default

        def test_fuzzy_recall_links_word_forms_but_not_short_words(self):
            self.assertEqual(fuzzy_recall({"argentinian"}, {"argentin"}), 1.0)
            self.assertEqual(overlap_recall({"argentinian"}, {"argentin"}), 0.0)
            self.assertEqual(fuzzy_recall({"art"}, {"arts"}), 0.0)
            passages = [{"title": "Florencia Bertotti",
                         "text": "<s>Florencia Bertotti is an Argentine actress and singer.</s>"}]
            fact = "Florencia Bertotti is an Argentinian."
            exact = EntailmentFactChecker(_FakeEntailment("Argentine"), fact_prune_threshold=0.3,
                                          prune_metric="recall")
            fuzzy = EntailmentFactChecker(_FakeEntailment("Argentine"), fact_prune_threshold=0.3,
                                          prune_metric="fuzzy_recall")
            self.assertEqual(exact.predict(fact, passages), "NS")             # pruned: different word form
            self.assertEqual(fuzzy.predict(fact, passages), "S")              # kept and entailed

        def test_empty_inputs_are_not_supported(self):
            checker = EntailmentFactChecker(_FakeEntailment(), fact_prune_threshold=0.0)
            self.assertEqual(checker.predict("Alice sang.", []), "NS")
            self.assertEqual(checker.predict("   ", self.PASSAGES), "NS")
            self.assertEqual(checker.predict("Alice sang.", [{"title": "Alice", "text": "<s></s>"}]), "NS")


    @unittest.skipUnless(os.path.exists(DATA), "data/ not present")
    class DriverSmokeTest(unittest.TestCase):
        def test_entailment_mode_runs_through_the_unmodified_driver(self):
            """ Runs factchecking_main.py --mode entailment end to end with the tiny model in place of the download. """
            import io
            import runpy
            import sys
            from contextlib import redirect_stdout
            import transformers
            tokenizer = _tiny_tokenizer(max_length=128)
            model = _tiny_model(tokenizer)
            saved = (transformers.AutoTokenizer.from_pretrained,
                     transformers.AutoModelForSequenceClassification.from_pretrained, sys.argv)
            transformers.AutoTokenizer.from_pretrained = lambda *a, **k: tokenizer
            transformers.AutoModelForSequenceClassification.from_pretrained = lambda *a, **k: model
            sys.argv = ["factchecking_main.py", "--mode", "entailment"]
            out = io.StringIO()
            try:
                cwd = os.getcwd()
                os.chdir(ROOT)
                with redirect_stdout(out):
                    runpy.run_path(os.path.join(ROOT, "factchecking_main.py"), run_name="__main__")
            finally:
                os.chdir(cwd)
                (transformers.AutoTokenizer.from_pretrained,
                 transformers.AutoModelForSequenceClassification.from_pretrained, sys.argv) = saved
            self.assertIn("Accuracy:", out.getvalue())

    suite = unittest.TestSuite()
    @unittest.skipUnless(os.path.exists(DATA), "data/ not present")
    class CompareSmokeTest(unittest.TestCase):
        def test_compare_runs_with_and_without_pruning(self):
            """ Runs  python factcheck_experiments.py compare  on a few facts with the tiny model in place of the download. """
            import io
            from contextlib import redirect_stdout
            import transformers
            tokenizer = _tiny_tokenizer(max_length=128)
            model = _tiny_model(tokenizer)
            saved = (transformers.AutoTokenizer.from_pretrained,
                     transformers.AutoModelForSequenceClassification.from_pretrained)
            transformers.AutoTokenizer.from_pretrained = lambda *a, **k: tokenizer
            transformers.AutoModelForSequenceClassification.from_pretrained = lambda *a, **k: model
            out_dir = tempfile.mkdtemp()
            buffer = io.StringIO()
            cwd = os.getcwd()
            try:
                with redirect_stdout(buffer):
                    code = _compare_pruning(["--max_facts", "6", "--out_dir", out_dir])
            finally:
                os.chdir(cwd)
                (transformers.AutoTokenizer.from_pretrained,
                 transformers.AutoModelForSequenceClassification.from_pretrained) = saved
            self.assertEqual(code, 0)
            self.assertEqual(buffer.getvalue().count("Accuracy: "), 2)
            report = open(os.path.join(out_dir, "compare_pruning.md"), encoding="utf-8").read()
            self.assertIn("WITH pruning", report)
            self.assertIn("WITHOUT pruning", report)

    class SequenceMetricTests(unittest.TestCase):
        def test_rouge_and_bleu_match_the_reference_libraries(self):
            """ Hand-written ROUGE-2 / ROUGE-L recall and BLEU-4 against rouge_score and NLTK on random sequences. """
            import random as _random
            from rouge_score import rouge_scorer
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
            scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=False)
            smooth = SmoothingFunction().method1
            rng = _random.Random(0)
            vocab = ["w%d" % i for i in range(10)]
            for _ in range(500):
                fact = [rng.choice(vocab) for _ in range(rng.randint(1, 9))]
                passage = [rng.choice(vocab) for _ in range(rng.randint(1, 40))]
                ref = scorer.score(" ".join(fact), " ".join(passage))
                self.assertAlmostEqual(rouge_n_recall(fact, passage, 2), ref["rouge2"].recall, places=12)
                self.assertAlmostEqual(rouge_l_recall(fact, passage), ref["rougeL"].recall, places=12)
                self.assertAlmostEqual(bleu(fact, passage), sentence_bleu([passage], fact, smoothing_function=smooth),
                                       places=12)

        def test_sequence_metrics_respect_word_order(self):
            self.assertEqual(rouge_n_recall(["born", "1950"], ["1950", "born"], 2), 0.0)
            self.assertEqual(overlap_recall(["born", "1950"], ["1950", "born"]), 1.0)
            self.assertEqual(rouge_l_recall(["a", "b", "c"], ["a", "x", "c", "b"]), 2 / 3)
            self.assertEqual(bleu([], ["a"]), 0.0)
            self.assertEqual(rouge_n_recall(["a"], ["a", "b"], 2), 0.0)  # no bigrams in a one-word fact

        def test_word_overlap_score_accepts_the_new_metrics(self):
            pre = TextPreprocessor()
            passages = [{"title": "Alice Smith", "text": "<s>Alice Smith was born in Paris in 1950.</s>"}]
            for metric in ("rouge2", "rougeL", "bleu"):
                value = word_overlap_score("Alice Smith was born in Paris.", passages, pre, metric, "max", True)
                self.assertTrue(0.0 < value <= 1.0, metric)
            with self.assertRaises(ValueError):
                word_overlap_score("x", passages, pre, "meteor")

    @unittest.skipUnless(os.path.exists(DATA), "data/ not present")
    class CommandSmokeTests(unittest.TestCase):
        def _with_tiny_model(self, function, argv):
            import io
            from contextlib import redirect_stdout
            import transformers
            tokenizer = _tiny_tokenizer(max_length=128)
            model = _tiny_model(tokenizer)
            saved = (transformers.AutoTokenizer.from_pretrained,
                     transformers.AutoModelForSequenceClassification.from_pretrained)
            transformers.AutoTokenizer.from_pretrained = lambda *a, **k: tokenizer
            transformers.AutoModelForSequenceClassification.from_pretrained = lambda *a, **k: model
            buffer, cwd = io.StringIO(), os.getcwd()
            try:
                with redirect_stdout(buffer):
                    code = function(argv)
            finally:
                os.chdir(cwd)
                (transformers.AutoTokenizer.from_pretrained,
                 transformers.AutoModelForSequenceClassification.from_pretrained) = saved
            return code, buffer.getvalue()

        def test_part1_reproduces_the_graded_word_overlap_result(self):
            out = tempfile.mkdtemp()
            code, _ = self._with_tiny_model(_part1_main, ["--out_dir", out, "--fig_dir", out, "--bootstrap", "20",
                                                          "--cv_repeats", "1"])
            self.assertEqual(code, 0)
            summary = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
            self.assertIn("Accuracy 184/221", summary)   # same as python factchecking_main.py --mode word_overlap
            self.assertIn("BLEU-4", summary)

        def test_error_worksheet_lists_every_error(self):
            out = tempfile.mkdtemp()
            code, printed = self._with_tiny_model(_error_worksheet, [
                "--max_facts", "12", "--cache", os.path.join(out, "units.jsonl"), "--out_dir", out, "--per_type", "2"])
            self.assertEqual(code, 0)
            with open(os.path.join(out, "error_worksheet.csv"), encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            line = next(l for l in printed.splitlines() if "[TP FN; FP TN] = [" in l)
            tp, fn, fp, tn = (int(x) for x in re.findall(r"-?\d+", line.split("= [")[-1]))
            self.assertEqual(sum(r["type"] == "FP" for r in rows), fp)
            self.assertEqual(sum(r["type"] == "FN" for r in rows), fn)
            self.assertLessEqual(sum(r["selected"] == "True" for r in rows), 4)
            self.assertTrue(all("category" in r and "evidence_1" in r for r in rows))

    for case in (SequenceMetricTests, EntailmentModelTests, EntailmentFactCheckerTests, DriverSmokeTest,
                 CompareSmokeTest, CommandSmokeTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _cli():
    """ python factcheck_experiments.py {part1, experiments, compare, errors, selftest} [options]; see each command's --help. """
    commands = {"part1": _part1_main, "experiments": _x_main, "compare": _compare_pruning,
                "errors": _error_worksheet, "selftest": _self_test}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("usage: python factcheck_experiments.py {part1,experiments,compare,errors,selftest} [options]\n"
              "The graded commands are run through factchecking_main.py, e.g.\n"
              "  python factchecking_main.py --mode entailment")
        sys.exit(2)
    sys.exit(commands[sys.argv[1]](sys.argv[2:]) or 0)


if __name__ == "__main__":
    _cli()
