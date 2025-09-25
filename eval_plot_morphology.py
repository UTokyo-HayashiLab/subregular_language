#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_plot_morphology.py
- Evaluates the saved logistic regression on test set.
- Adds *logistic-only* separability diagnostics on test:
  (A) Normalized margin statistics and histogram
Saves:
  - metrics_test.json  (includes margin stats)
  - roc.png, pr.png, confusion.png, top_predicates.png (existing)
  - margin_hist.png
"""
import os, csv, argparse, json, pickle
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# ---------- IO & features (same as train) ----------
def read_csv(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            seq = row["affix_seq"].split(";") if row["affix_seq"] else []
            rows.append({"word": row["word"], "seq": seq, "y": int(row["label"])})
    return rows

def subseqs_upto_m(seq: List[str], m: int):
    S=set()
    n=len(seq)
    for L in range(1, m+1):
        def rec(start, l, pref):
            if l==0:
                S.add(tuple(pref)); return
            for i in range(start, n-l+1):
                pref.append(seq[i]); rec(i+1, l-1, pref); pref.pop()
        rec(0, L, [])
    return S

def kgrams_with_boundaries(seq: List[str], k: int):
    BOS, EOS = "<#>", "</#>"
    t = [BOS]*(k-1) + seq + [EOS]*(k-1)
    return [tuple(t[i:i+k]) for i in range(len(t)-k+1)]

def vectorize(rows, PT_basis, LTT_basis, use_PT=True, use_LTT=True, m=2, k=2):
    feats=[]
    for row in rows:
        seq = row["seq"]
        v=[]
        if use_PT:
            S = subseqs_upto_m(seq, m)
            for tpl in PT_basis:
                v.append(1 if tuple(tpl) in S else 0)
        if use_LTT:
            memo_counts={}
            maxL = max((len(b["pat"]) for b in LTT_basis), default=0)
            for L in range(1, maxL+1):
                grams = kgrams_with_boundaries(seq, L)
                memo_counts[L]={}
                for g in grams:
                    memo_counts[L][tuple(g)] = memo_counts[L].get(tuple(g), 0)+1
            for b in LTT_basis:
                pat = tuple(b["pat"]); thr = b["thr"]
                c = memo_counts[len(pat)].get(pat, 0)
                v.append(1 if c >= thr else 0)
        feats.append(v)
    X = np.array(feats, dtype=np.int8) if feats else np.zeros((0,0), dtype=np.int8)
    y = np.array([row["y"] for row in rows], dtype=np.int64)
    return X, y

# ---------- Diagnostics ----------
def normalized_margin_stats(clf, X, y):
    if X.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    w = clf.coef_.ravel()
    b = clf.intercept_[0] if getattr(clf, "fit_intercept", True) else 0.0
    norm = np.linalg.norm(w) + 1e-12
    ypm1 = 2*y - 1
    scores = (X @ w + b) / norm
    margins = ypm1 * scores
    qs = np.percentile(margins, [5,25,50,75,95])
    return {
        "min": float(margins.min()),
        "mean": float(margins.mean()),
        "std": float(margins.std()),
        "p5": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
    }, margins

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test = read_csv(os.path.join(args.data_dir, "test.csv"))
    with open(os.path.join(args.run_dir, "PT_basis.json"), "r") as f:
        PT_basis = json.load(f)
    with open(os.path.join(args.run_dir, "LTT_basis.json"), "r") as f:
        LTT_basis = json.load(f)
    with open(os.path.join(args.run_dir, "clf.pkl"), "rb") as f:
        clf = pickle.load(f)

    use_PT  = len(PT_basis)  > 0
    use_LTT = len(LTT_basis) > 0
    Xte, yte = vectorize(test, PT_basis, LTT_basis, use_PT, use_LTT, m=args.m, k=args.k)

    prob = clf.predict_proba(Xte)[:,1]
    yhat = (prob>=0.5).astype(int)

    acc = accuracy_score(yte, yhat)
    f1  = f1_score(yte, yhat)
    auc = roc_auc_score(yte, prob)

    # (A) normalized margin stats on test
    mstats, margins = normalized_margin_stats(clf, Xte, yte)

    metrics = {
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "n_test": int(len(yte)),
        "dim": int(Xte.shape[1]),
        "margin_stats": mstats
    }

    # Save metrics
    import json as _json
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w") as f:
        _json.dump(metrics, f, indent=2)
    print(metrics)

    # ROC & PR curves
    fpr, tpr, _ = roc_curve(yte, prob)
    prec, rec, _ = precision_recall_curve(yte, prob)

    plt.figure(); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "roc.png")); plt.close()

    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "pr.png")); plt.close()

    # Confusion matrix
    cm = confusion_matrix(yte, yhat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ill-formed","well-formed"])
    disp.plot(values_format="d"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "confusion.png")); plt.close()

    # Top feature weights
    coef = clf.coef_[0]
    names=[]
    if use_PT:
        names += [f"PT:{'-'.join(map(str, tpl))}" for tpl in PT_basis]
    if use_LTT:
        names += [f"LTT:{'-'.join(map(str, b['pat']))}>= {b['thr']}" for b in LTT_basis]
    coef_names = list(zip(coef, names))
    coef_sorted = sorted(coef_names, key=lambda x: abs(x[0]), reverse=True)[:30]
    vals = [c for c,_ in coef_sorted]
    lbls = [n for _,n in coef_sorted]
    plt.figure(figsize=(10,6)); plt.barh(range(len(vals)), vals); plt.yticks(range(len(vals)), lbls)
    plt.gca().invert_yaxis()
    plt.xlabel("Weight"); plt.title("Top deciding predicates (by |weight|)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "top_predicates.png")); plt.close()

    # Margin histogram
    plt.figure(figsize=(6,4))
    plt.hist(margins, bins=40)
    plt.xlabel("Normalized margin  y*(w·x+b)/||w||")
    plt.ylabel("Count")
    plt.title("Margin distribution (test)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "margin_hist.png")); plt.close()

if __name__ == "__main__":
    main()

