#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download MorphoLex-en (English derivational morphology) Excel file.

Usage:
  python download_morpholex_en.py --out Morpholex_en.xlsx
"""
import argparse, os, sys, urllib.request

URLS = [
    # Official GitHub repo for MorphoLex-en (English)
    "https://raw.githubusercontent.com/hugomailhot/MorphoLex-en/master/MorphoLEX_en.xlsx",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="MorphoLEX_en.xlsx")
    args = ap.parse_args()

    for url in URLS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, args.out)
            print(f"Saved to: {args.out}")
            return
        except Exception as e:
            print(f"[WARN] Failed: {e}")
    print("[ERROR] All downloads failed. Please download the Excel manually and pass its path.")
    sys.exit(1)

if __name__ == "__main__":
    main()

