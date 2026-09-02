"""
Prepare a base GWAS for the PRS-Net pipeline: validate the column contract and
append BETA = log(OR) as column 12.

Input  : CHR BP SNP A1 A2 N SE P OR INFO MAF          (11 columns, tab-separated)
Output : CHR BP SNP A1 A2 N SE P OR INFO MAF BETA     (12 columns, tab-separated)

The first 11 columns are already in the order the pipeline expects, so this only
appends. Nothing is reordered.

Why this is a script and not a one-line awk
-------------------------------------------
log(OR) is trivial; the failure modes are not, and every one of them is silent
downstream:
  * an "OR" column that already holds betas -> log() of a near-zero number
  * OR <= 0 or NA -> -inf / nan betas that PLINK scores as zero without complaint
  * A1 not actually the effect allele -> every beta sign-flipped, model trains fine
  * a 'chr' prefix on CHR -> the harmonisation awk merges on (CHR, BP) against your
    .bim and matches nothing
  * INFO/MAF outside [0,1] -> step 1's filters silently drop everything

Each is checked and reported rather than left to surface as a bad AUROC.
"""

import argparse
import sys

import numpy as np
import pandas as pd

REQUIRED_IN = ["CHR", "BP", "SNP", "A1", "A2", "N", "SE", "P", "OR", "INFO", "MAF"]
REQUIRED_OUT = REQUIRED_IN + ["BETA"]
VALID_ALLELES = {"A", "C", "G", "T"}


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--effect_allele", default="A1", choices=["A1", "A2"],
                    help="Which allele the OR is expressed with respect to. If A2, "
                         "BETA is negated so that A1 becomes the effect allele, "
                         "which is what --score assumes.")
    ap.add_argument("--strip_chr_prefix", action="store_true",
                    help="Remove a leading 'chr' from CHR to match a PLINK .bim")
    ap.add_argument("--drop_invalid", action="store_true",
                    help="Drop rows with unusable OR/INFO/MAF instead of failing")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, dtype=str)
    n_in = len(df)
    print(f"Read {n_in:,} rows, columns: {list(df.columns)}")

    # ---- column contract ---------------------------------------------------
    missing = [c for c in REQUIRED_IN if c not in df.columns]
    if missing:
        fail(f"missing required columns: {missing}")
    if list(df.columns[:11]) != REQUIRED_IN:
        print(f"NOTE: reordering first 11 columns to {REQUIRED_IN}")
    if "BETA" in df.columns:
        fail("input already has a BETA column -- use it directly rather than "
             "recomputing from OR")
    df = df[REQUIRED_IN].copy()

    # ---- numeric coercion --------------------------------------------------
    for col in ["BP", "N", "SE", "P", "OR", "INFO", "MAF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- is the OR column actually an OR? ----------------------------------
    orv = df["OR"].dropna()
    if len(orv) == 0:
        fail("OR column is entirely non-numeric")
    med, frac_neg = float(orv.median()), float((orv <= 0).mean())
    print(f"OR: median={med:.4f}  min={orv.min():.4g}  max={orv.max():.4g}  "
          f"frac<=0={frac_neg:.4%}")
    if abs(med) < 0.5:
        fail(f"OR median is {med:.4f}, which is close to 0 rather than 1. This "
             "column looks like it already holds BETA (log-odds), not OR. Rename "
             "it to BETA and skip this script.")
    if not (0.8 < med < 1.25):
        print(f"WARNING: OR median {med:.4f} is far from 1.0. Verify the column.",
              file=sys.stderr)

    # ---- CHR convention ----------------------------------------------------
    chr_has_prefix = df["CHR"].astype(str).str.lower().str.startswith("chr").any()
    if chr_has_prefix:
        if args.strip_chr_prefix:
            df["CHR"] = df["CHR"].astype(str).str.replace(r"^chr", "", case=False, regex=True)
            print("Stripped 'chr' prefix from CHR")
        else:
            print("WARNING: CHR values carry a 'chr' prefix. PLINK .bim files use bare "
                  "numbers, and the harmonisation step merges on (CHR, BP) -- it will "
                  "match nothing. Re-run with --strip_chr_prefix if your .bim is bare.",
                  file=sys.stderr)

    # ---- alleles -----------------------------------------------------------
    for col in ["A1", "A2"]:
        df[col] = df[col].astype(str).str.upper().str.strip()
    bad_allele = ~(df["A1"].isin(VALID_ALLELES) & df["A2"].isin(VALID_ALLELES))
    if bad_allele.any():
        print(f"WARNING: {int(bad_allele.sum()):,} rows have non-SNP alleles "
              "(indels/multi-char). Step 1 does not remove these explicitly; they "
              "will simply fail to match the .bim.", file=sys.stderr)

    # ---- validity masks ----------------------------------------------------
    bad = {
        "OR missing or <= 0": df["OR"].isna() | (df["OR"] <= 0),
        "P missing or outside (0, 1]": df["P"].isna() | (df["P"] <= 0) | (df["P"] > 1),
        "INFO missing or outside [0, 1]": df["INFO"].isna() | (df["INFO"] < 0) | (df["INFO"] > 1),
        "MAF missing or outside [0, 0.5]": df["MAF"].isna() | (df["MAF"] < 0) | (df["MAF"] > 0.5),
        "BP missing": df["BP"].isna(),
    }
    drop = np.zeros(len(df), dtype=bool)
    for label, mask in bad.items():
        n = int(mask.sum())
        if n:
            print(f"  {label}: {n:,} rows")
        drop |= mask.values

    n_drop = int(drop.sum())
    if n_drop:
        if not args.drop_invalid:
            fail(f"{n_drop:,} rows are unusable. Inspect the counts above, then "
                 "re-run with --drop_invalid to remove them.")
        df = df[~drop].copy()
        print(f"Dropped {n_drop:,} rows ({n_drop / n_in:.2%})")

    if df["MAF"].max() > 0.5:
        print("WARNING: MAF > 0.5 present -- column may hold effect-allele frequency "
              "rather than minor-allele frequency. Step 1 filters on MAF > threshold, "
              "so this only affects which rare variants survive.", file=sys.stderr)

    # ---- BETA --------------------------------------------------------------
    df["BETA"] = np.log(df["OR"].values)
    if args.effect_allele == "A2":
        df["BETA"] = -df["BETA"]
        df["OR"] = 1.0 / df["OR"]
        df[["A1", "A2"]] = df[["A2", "A1"]].values
        print("Effect allele was A2: negated BETA, inverted OR, swapped A1/A2 so that "
              "A1 is the effect allele (what --score assumes).")

    df = df[REQUIRED_OUT]
    df.to_csv(args.output, sep="\t", index=False, na_rep="NA")

    print(f"\nWrote {len(df):,} rows to {args.output}")
    print(f"BETA: mean={df['BETA'].mean():.5f}  sd={df['BETA'].std():.5f}  "
          f"frac>0={float((df['BETA'] > 0).mean()):.2%}")
    print("(frac>0 near 50% is expected genome-wide; a strong skew suggests the "
          "effect-allele convention is wrong.)")
    print("\nColumn order for --score: 3=SNP, 4=A1, 12=BETA. "
          "SE (col 7) is never read by this pipeline.")


if __name__ == "__main__":
    main()
