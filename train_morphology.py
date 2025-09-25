#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_morphology.py
- Trains logistic regression on PT (subsequence) and/or LTT (k-gram threshold) features.
- Adds *logistic-only* separability diagnostics:
  (A) Normalized margin statistics on dev set
  (B) Regularization-path sweep over C: ||w||, train log-loss, train accuracy

Outputs in --out_dir:
  - clf.pkl, vocab.json, PT_basis.json, LTT_basis.json
  - metrics_dev.json  (includes margin stats)
  - path_sweep.csv    (C, coef_norm, train_logloss, train_acc)
  - path_coefnorm_vs_C.png, path_logloss_vs_C.png
"""
import os, csv, argparse, json, itertools, pickle
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

# ---------- IO ----------
def read_csv(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            seq = row["affix_seq"].split(";") if row["affix_seq"] else []
            rows.append({"word": row["word"], "seq": seq, "y": int(row["label"])})
    return rows

# ---------- Bases ----------
def build_PT_basis(vocab: List[str], m: int):
    import itertools
    B=[]
    for L in range(1, m+1):
        for tpl in itertools.product(vocab, repeat=L):
            B.append(("PT", tpl))
    return B

def build_LTT_basis(vocab: List[str], k: int, tau_uni=2, tau_bi=1):
    BOS, EOS = "<#>", "</#>"
    import itertools
    Pats=[]
    sym = [BOS] + vocab + [EOS]
    for L in range(1, k+1):
        Pats += list(itertools.product(sym, repeat=L))
    basis=[]
    for pat in Pats:
        T = tau_uni if len(pat)==1 else tau_bi
        for t in range(1, T+1):
            basis.append(("LTT", pat, t))
    return basis

# ---------- Feature maps ----------
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
            for (_tag, tpl) in PT_basis:
                v.append(1 if tpl in S else 0)
        if use_LTT:
            memo_counts={}
            # pre-count for all lengths present in basis
            lengths=set(len(pat) for (_tag, pat, thr) in LTT_basis) if LTT_basis else set()
            for L in sorted(lengths):
                grams = kgrams_with_boundaries(seq, L)
                memo_counts[L]={}
                for g in grams:
                    memo_counts[L][g] = memo_counts[L].get(g, 0)+1
            for (_tag, pat, thr) in LTT_basis:
                c = memo_counts[len(pat)].get(pat, 0)
                v.append(1 if c >= thr else 0)
        feats.append(v)
    X = np.array(feats, dtype=np.int8) if feats else np.zeros((0,0), dtype=np.int8)
    y = np.array([row["y"] for row in rows], dtype=np.int64)
    return X, y

# ---------- Diagnostics ----------
def normalized_margin_stats(clf: LogisticRegression, X: np.ndarray, y: np.ndarray):
    """Compute margin stats with ||w||2 normalization.
       y in {0,1} -> map to {-1,+1}."""
    if X.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    w = clf.coef_.ravel()
    b = clf.intercept_[0] if clf.fit_intercept else 0.0
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
    }

def sweep_over_C(Xtr, ytr, C_list, seed):
    rows=[]
    for C in C_list:
        clf = LogisticRegression(max_iter=1000, C=C, random_state=seed)
        clf.fit(Xtr, ytr)
        w = clf.coef_.ravel()
        coef_norm = float(np.linalg.norm(w))
        # train metrics
        prob_tr = clf.predict_proba(Xtr)[:,1]
        acc_tr = float((prob_tr>=0.5).astype(int).mean())
        ll_tr  = float(log_loss(ytr, prob_tr, labels=[0,1]))
        rows.append({"C": C, "coef_norm": coef_norm, "train_logloss": ll_tr, "train_acc": acc_tr})
    return rows

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--features", nargs="+", choices=["PT","LTT"], default=["PT","LTT"])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20250922)
    ap.add_argument("--path_Cs", type=str, default="0.1,1,10,100,1000",
                    help="Comma-separated Cs for regularization-path diagnostics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = read_csv(os.path.join(args.data_dir, "train.csv"))
    dev   = read_csv(os.path.join(args.data_dir, "dev.csv"))

    # vocab & bases
    vocab = sorted(list({a for r in train for a in r["seq"]}))
    use_PT = "PT" in args.features
    use_LTT = "LTT" in args.features
    PT_basis  = build_PT_basis(vocab, args.m) if use_PT else []
    LTT_basis = build_LTT_basis(vocab, args.k) if use_LTT else []

    Xtr, ytr = vectorize(train, PT_basis, LTT_basis, use_PT, use_LTT, m=args.m, k=args.k)
    Xdv, ydv = vectorize(dev,   PT_basis, LTT_basis, use_PT, use_LTT, m=args.m, k=args.k)

    # Logistic Regression (main model)
    clf = LogisticRegression(max_iter=1000, C=args.C, random_state=args.seed)
    clf.fit(Xtr, ytr)
    p_dv   = clf.predict_proba(Xdv)[:,1]
    yhat_dv= (p_dv>=0.5).astype(int)

    metrics = {
        "acc": float(accuracy_score(ydv, yhat_dv)),
        "f1":  float(f1_score(ydv, yhat_dv)),
        "auc": float(roc_auc_score(ydv, p_dv)),
        "n_train": int(len(train)),
        "n_dev": int(len(dev)),
        "dim": int(Xtr.shape[1])
    }

    # (A) normalized margin stats on dev
    margin_stats = normalized_margin_stats(clf, Xdv, ydv)
    metrics["margin_stats"] = margin_stats

    # (B) regularization path over C (train set)
    C_list = [float(s) for s in args.path_Cs.split(",") if s.strip()]
    path_rows = sweep_over_C(Xtr, ytr, C_list, args.seed)

    # Save runs
    with open(os.path.join(args.out_dir, "metrics_dev.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(args.out_dir, "PT_basis.json"), "w") as f:
        json.dump([list(tpl) for (_tag, tpl) in PT_basis], f)
    with open(os.path.join(args.out_dir, "LTT_basis.json"), "w") as f:
        json.dump([{"pat": list(pat), "thr": thr} for (_tag, pat, thr) in LTT_basis], f)
    with open(os.path.join(args.out_dir, "clf.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # Write path CSV
    path_csv = os.path.join(args.out_dir, "path_sweep.csv")
    import csv as _csv
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["C","coef_norm","train_logloss","train_acc"])
        for r in path_rows:
            w.writerow([r["C"], f"{r['coef_norm']:.6f}", f"{r['train_logloss']:.6f}", f"{r['train_acc']:.6f}"])

    # Plots for path sweep
    Cs = [r["C"] for r in path_rows]
    norms = [r["coef_norm"] for r in path_rows]
    lls = [r["train_logloss"] for r in path_rows]

    plt.figure(); plt.plot(Cs, norms, marker="o")
    plt.xscale("log"); plt.xlabel("C"); plt.ylabel("||w||2")
    plt.title("Coefficient norm vs C (logistic)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "path_coefnorm_vs_C.png")); plt.close()

    plt.figure(); plt.plot(Cs, lls, marker="o")
    plt.xscale("log"); plt.xlabel("C"); plt.ylabel("Train log-loss")
    plt.title("Train log-loss vs C (logistic)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "path_logloss_vs_C.png")); plt.close()

    print("Dev metrics (with separability diagnostics) saved to:", args.out_dir)

if __name__ == "__main__":
    main()

