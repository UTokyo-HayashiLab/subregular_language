#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust converter for MorphoLex-en Excel -> CSV (word, affix_seq)

- Scans all sheets.
- Extracts PREFIX and SUFFIX inventories from sheets named like "All prefixes" / "All suffixes".
- Collects candidate WORDS by scanning every cell in the remaining sheets
  (regex filter for "word-like" strings).
- For each word, builds an ordered affix sequence:
    [prefixes...] + [suffixes...]
  using the extracted inventories (longest-first greedy peel).
- Keeps rows with min_len <= len(seq) <= max_len.

Usage:
  python morpholex_to_affixcsv_v2.py --xlsx Morpholex_en.xlsx --out morpholex_affixes.csv \
      --min_len 1 --max_len 6

Tips:
  - Add --include_sheets '0-1-0,0-1-1,...' to restrict scanning if needed.
  - Use --exclude_sheets 'Presentation,All roots' to skip non-lexical sheets.
"""

import argparse, os, sys, re, pandas as pd

WORD_RX = re.compile(r"^[A-Za-z][A-Za-z'\-]{2,}$")  # simple "word-like" filter

def clean_str(x):
    if x is None: return ""
    s = str(x).strip()
    return s

def harvest_list_from_sheet(df):
    # pick 1D list from first column (or any non-empty string cells)
    vals = []
    # prefer first column if it looks dense
    col0 = df.columns[0]
    col0_vals = [clean_str(v) for v in df[col0].tolist()]
    dens0 = sum(1 for v in col0_vals if v)
    if dens0 >= max(10, int(0.2*len(col0_vals))):
        vals = [v for v in col0_vals if v]
    else:
        # fallback: scan all cells
        for c in df.columns:
            for v in df[c].tolist():
                v = clean_str(v)
                if v:
                    vals.append(v)
    # de-dup, keep order
    seen=set(); out=[]
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def extract_affix_inventories(xls):
    prefixes=set(); suffixes=set()
    for name in xls.sheet_names:
        low = name.lower()
        if "prefix" in low:
            df = pd.read_excel(xls, sheet_name=name)
            vals = harvest_list_from_sheet(df)
            for v in vals:
                tok = v.lower().strip("- ")
                if tok and tok.isalpha():
                    prefixes.add(tok)
        elif "suffix" in low:
            df = pd.read_excel(xls, sheet_name=name)
            vals = harvest_list_from_sheet(df)
            for v in vals:
                tok = v.lower().strip("- ")
                if tok and tok.isalpha():
                    suffixes.add(tok)
    # sort by length desc for greedy peel
    pref_list = sorted(prefixes, key=len, reverse=True)
    suff_list = sorted(suffixes, key=len, reverse=True)
    return pref_list, suff_list

def greedy_affix_sequence(word, pref_list, suff_list):
    w = re.sub(r"[^\w]", "", (word or "").lower())
    if not w: return []
    prefixes=[]; core=w
    # peel multiple prefixes (longest-first)
    for pref in pref_list:
        # allow nesting, but avoid over-stripping very short remainders
        while core.startswith(pref) and len(core) > len(pref) + 2:
            prefixes.append(pref)
            core = core[len(pref):]
    # peel multiple suffixes (longest-first)
    suffixes=[]; changed=True
    while changed:
        changed=False
        for suf in suff_list:
            if core.endswith(suf) and len(core) > len(suf) + 2:
                suffixes.append(suf)
                core = core[:-len(suf)]
                changed=True
                break
    return prefixes + list(reversed(suffixes))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--include_sheets", type=str, default=None,
                    help="Comma-separated sheet names to include (default: all)")
    ap.add_argument("--exclude_sheets", type=str, default="Presentation,All roots",
                    help="Comma-separated sheet names to skip")
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=6)
    args = ap.parse_args()

    if not os.path.isfile(args.xlsx):
        print(f"[ERROR] File not found: {args.xlsx}"); sys.exit(1)

    try:
        xls = pd.ExcelFile(args.xlsx)
    except Exception as e:
        print(f"[ERROR] Failed to read Excel: {e}"); sys.exit(1)

    include = None
    if args.include_sheets:
        include = set([s.strip() for s in args.include_sheets.split(",") if s.strip()])
    exclude = set([s.strip() for s in args.exclude_sheets.split(",") if s.strip()])

    # 1) affix inventories from “All prefixes / All suffixes”
    pref_list, suff_list = extract_affix_inventories(xls)
    if not pref_list or not suff_list:
        print("[WARN] Could not extract prefixes/suffixes from sheets; using small defaults.")
        pref_list = ["un","re","pre","dis","non","over","sub","inter","trans","mis","under"]
        suff_list = ["ness","less","ment","tion","sion","able","ible","ize","ise","ify","al","er","or","ly"]

    # 2) collect word candidates by scanning sheets
    words=set()
    for name in xls.sheet_names:
        if include and name not in include:
            continue
        if name in exclude:
            continue
        try:
            df = pd.read_excel(xls, sheet_name=name)
        except Exception:
            continue
        # scan all cells; pick "word-like" strings
        for c in df.columns:
            for v in df[c].tolist():
                s = clean_str(v)
                if s and WORD_RX.match(s):
                    # avoid blatantly affix-y items (often contain hyphens/spaces)
                    words.add(s.lower())
        # optional: small progress
        # print(name, "->", len(words))

    # 3) build sequences
    rows=[]
    for w in sorted(words):
        seq = greedy_affix_sequence(w, pref_list, suff_list)
        if args.min_len <= len(seq) <= args.max_len:
            rows.append((w, ";".join(seq)))

    # 4) write CSV
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["word","affix_seq"])
        for w, seq in rows:
            writer.writerow([w, seq])

    print(f"Sheets scanned: {len(xls.sheet_names)}")
    print(f"Words collected: {len(words)}")
    print(f"Rows kept (min_len={args.min_len}, max_len={args.max_len}): {len(rows)}")
    print(f"Wrote {args.out}")
    if rows[:5]:
        print("Example:", rows[:5])

if __name__ == "__main__":
    main()

