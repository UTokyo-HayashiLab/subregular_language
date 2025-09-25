#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_prep_morphology.py
- Downloads (optional) and prepares English derivational morphology data.
- Extracts affix sequences (prefixes then suffixes) from each word.
- Generates negative (ill-formed) sequences by controlled permutations/substitutions.
- Produces train/dev/test splits with matched length distributions.

Usage:
  python data_prep_morphology.py --input_csv path/to/MorphoLex_English.csv --out_dir data_out
Optional:
  --download_url URL  (if you have a direct URL for the CSV; otherwise place file locally)

Notes:
- If your CSV has columns like 'Word','Prefixes','Suffixes', they will be used.
- Otherwise a heuristic segmenter over common derivational affixes is applied.

Output files in out_dir:
  affixes.json       (prefix_set & suffix_set used; for reproducibility)
  dataset.csv        (word, affix_seq (semicolon-separated), label)
  train.csv / dev.csv / test.csv
"""
import os, csv, argparse, json, random, re, sys
from typing import List, Dict
from urllib.request import urlretrieve

# A compact list of common English derivational affixes (heuristic fallback)
PREFIXES = [
    "anti","auto","bi","co","counter","de","dis","en","em","fore","il","im","in","ir",
    "inter","mis","non","over","pre","re","semi","sub","super","trans","tri","ultra","under","uni","un"
]
SUFFIXES = [
    "ability","ibility","ization","isation","ment","ness","less","able","ible","tion","sion",
    "hood","ship","ward","wise","ful","ous","ish","ize","ise","ify","al","er","or","ly","en","ist","ity","ive","ic"
]

def safe_download(url: str, dest: str):
    try:
        print(f"Downloading: {url}")
        urlretrieve(url, dest)
        print(f"Saved to: {dest}")
        return True
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
        return False

def read_csv_rows(path: str):
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        sn = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = sn.sniff(sample) if sample else csv.excel
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = [r for r in reader]
        return rows

def heuristic_affix_sequence(word: str):
    w = word.lower()
    prefixes = []
    # greedily peel multiple prefixes (longest-first)
    for pref in sorted(PREFIXES, key=len, reverse=True):
        if w.startswith(pref) and len(w) > len(pref)+2:
            prefixes.append(pref)
            w = w[len(pref):]
    suffixes = []
    # greedily peel multiple suffixes (longest-first)
    changed = True
    while changed:
        changed = False
        for suf in sorted(SUFFIXES, key=len, reverse=True):
            if w.endswith(suf) and len(w) > len(suf)+2:
                suffixes.append(suf)
                w = w[:-len(suf)]
                changed = True
                break
    # affix sequence is prefixes (in order applied) followed by suffixes (in order applied)
    return prefixes + list(reversed(suffixes))

def extract_affix_sequence(row: Dict):
    # Prefer explicit columns if available
    word = None
    for key in row.keys():
        if key.lower() in ("word","orth","lemma","spelling"):
            word = row[key]
            break
    if word is None:
        # pick the first column value
        word = list(row.values())[0]
    # Columns for prefixes/suffixes if exist
    pref_cols = [k for k in row if "prefix" in k.lower()]
    suff_cols = [k for k in row if "suffix" in k.lower()]
    seq = []
    # collect ordered prefixes/suffixes if present
    if pref_cols:
        parts = []
        for k in pref_cols:
            parts += re.split(r"[;,+\s]+", str(row[k]).strip())
        seq += [p for p in parts if p]
    if suff_cols:
        parts = []
        for k in suff_cols:
            parts += re.split(r"[;,+\s]+", str(row[k]).strip())
        seq += [p for p in parts if p]
    if not seq:
        seq = heuristic_affix_sequence(word)
    return word, [a for a in seq if a]

def build_dataset(rows, max_len=6, min_len=1):
    pos = []
    affix_vocab = set()
    for r in rows:
        w, seq = extract_affix_sequence(r)
        if min_len <= len(seq) <= max_len:
            pos.append((w, seq))
            affix_vocab.update(seq)
    pos = list({(w, tuple(seq)) for w,seq in pos})  # deduplicate
    affix_vocab = sorted(list(affix_vocab))
    return pos, affix_vocab

def generate_negatives(pos_pairs, affix_vocab, neg_per_pos=1, seed=0):
    rnd = random.Random(seed)
    neg = []
    for w, seq in pos_pairs:
        L = len(seq)
        if L >= 2 and rnd.random() < 0.5:
            # permutation: swap two positions
            i, j = rnd.sample(range(L), 2)
            s2 = list(seq); s2[i], s2[j] = s2[j], s2[i]
        else:
            # substitution: replace one affix with a random different one
            i = rnd.randrange(L)
            candidates = [a for a in affix_vocab if a != seq[i]]
            if not candidates:
                continue
            s2 = list(seq); s2[i] = rnd.choice(candidates)
        if tuple(s2) != tuple(seq):
            neg.append((w, s2))
    return neg

def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=False, help="Path to MorphoLex-like CSV")
    ap.add_argument("--download_url", type=str, default=None, help="Direct URL to CSV (optional)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=20250922)
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="negatives per positive (≈)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = args.input_csv
    if src is None and args.download_url:
        dest = os.path.join(args.out_dir, "download.csv")
        ok = safe_download(args.download_url, dest)
        if not ok:
            print("[ERROR] Could not download. Please provide --input_csv.")
            sys.exit(1)
        src = dest
    if src is None:
        print("[ERROR] Provide --input_csv or --download_url")
        sys.exit(1)

    rows = read_csv_rows(src)
    pos_pairs, affix_vocab = build_dataset(rows)

    # negatives
    neg_pairs = generate_negatives(pos_pairs, affix_vocab, seed=args.seed)
    # balance
    rnd = random.Random(args.seed)
    rnd.shuffle(pos_pairs); rnd.shuffle(neg_pairs)
    if args.neg_ratio is not None:
        target_neg = int(len(pos_pairs) * args.neg_ratio)
        neg_pairs = neg_pairs[:target_neg]

    # Build flat rows
    data = []
    for w, seq in pos_pairs:
        data.append([w, ";".join(seq), 1])
    for w, seq in neg_pairs:
        data.append([w, ";".join(seq), 0])

    # Split by word (avoid leakage)
    rnd.shuffle(data)
    words = list({r[0] for r in data})
    rnd.shuffle(words)
    n = len(words); n_tr = int(0.7*n); n_dev = int(0.15*n)
    tr_words = set(words[:n_tr])
    dev_words = set(words[n_tr:n_tr+n_dev])
    te_words = set(words[n_tr+n_dev:])

    train = [r for r in data if r[0] in tr_words]
    dev   = [r for r in data if r[0] in dev_words]
    test  = [r for r in data if r[0] in te_words]

    write_csv(os.path.join(args.out_dir, "dataset.csv"), data, ["word","affix_seq","label"])
    write_csv(os.path.join(args.out_dir, "train.csv"), train, ["word","affix_seq","label"])
    write_csv(os.path.join(args.out_dir, "dev.csv"),   dev,   ["word","affix_seq","label"])
    write_csv(os.path.join(args.out_dir, "test.csv"),  test,  ["word","affix_seq","label"])

    meta = {"prefixes": PREFIXES, "suffixes": SUFFIXES}
    with open(os.path.join(args.out_dir, "affixes.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out_dir}/train.csv, dev.csv, test.csv")
    print(f"Positives: {len(pos_pairs)}  Negatives: {len(neg_pairs)}")
    print(f"Affix vocab size: {len(affix_vocab)}")

if __name__ == "__main__":
    main()

