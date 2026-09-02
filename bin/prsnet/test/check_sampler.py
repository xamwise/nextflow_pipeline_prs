#!/usr/bin/env python
"""
Confirm the sampler_num_samples patch took effect.

Run once from the repo root, before committing to a long training run:

    python bin/prsnet/test/check_sampler.py

It locates the pipeline artifacts, then prints steps-per-epoch for three
configurations. Nothing here belongs in the pipeline -- it is a one-off check.

Expected with the patch applied:

    no sampler                        ->   2 steps/epoch
    balanced, num_samples=None        ->   2 steps/epoch
    balanced, num_samples=51200       -> 100 steps/epoch     <- what you want
      x max_epochs 195               -> 19,500 total steps

Unpatched, the third line also prints 2, and `max_epochs: 195` would give 390
steps instead of 19,500 -- a 50x shortfall that looks like the model failing to
learn rather than a configuration problem.
"""

import argparse
import glob
import os
import sys


def find_one(patterns, label):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    sys.exit(f"Could not locate {label}. Tried:\n  " + "\n  ".join(patterns)
             + f"\nPass it explicitly: --{label} <path>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", default="bin", help="directory holding genotype_dataset.py")
    ap.add_argument("--h5", default=None)
    ap.add_argument("--pheno", default=None)
    ap.add_argument("--indices", default=None)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_samples", type=int, default=51200)
    ap.add_argument("--max_epochs", type=int, default=195)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.bin_dir))

    h5 = args.h5 or find_one(
        ["out/**/genotype_data.h5", "**/prsnet/features/genotype_data.h5"], "h5")
    pheno = args.pheno or find_one(
        [os.path.join(os.path.dirname(h5), "phenotypes.csv"),
         "out/**/phenotypes.csv"], "pheno")
    indices = args.indices or find_one(
        ["out/**/split_indices.npz", "out/**/*indices*.npz", "**/split_indices.npz"],
        "indices")

    print(f"h5      : {h5}")
    print(f"pheno   : {pheno}")
    print(f"indices : {indices}\n")

    from genotype_dataset import GenotypeDataModule

    import inspect
    sig = inspect.signature(GenotypeDataModule.__init__).parameters
    missing = [k for k in ("balanced_sampling", "sampler_num_samples")
               if k not in sig]
    if missing:
        sys.exit(
            f"GenotypeDataModule.__init__ has no {missing} parameter(s).\n"
            "The patch is not applied to the file being imported. Check that\n"
            f"  {os.path.abspath(os.path.join(args.bin_dir, 'genotype_dataset.py'))}\n"
            "is the file you edited."
        )

    def steps(**kw):
        dm = GenotypeDataModule(h5_file=h5, phenotype_file=pheno,
                               indices_file=indices, augment_train=False, **kw)
        n = len(dm.train_dataloader(fold=args.fold))
        ntr = (len(dm.fold_indices[args.fold]["train"])
               if dm.fold_indices else len(dm.train_indices))
        return n, ntr

    n0, ntr = steps(batch_size=args.batch_size)
    print(f"train fold {args.fold}: {ntr} samples, batch {args.batch_size}\n")
    print(f"  no sampler                       -> {n0:>4} steps/epoch")
    n1, _ = steps(batch_size=args.batch_size, balanced_sampling=True)
    print(f"  balanced, num_samples=None       -> {n1:>4} steps/epoch")
    n2, _ = steps(batch_size=args.batch_size, balanced_sampling=True,
                  sampler_num_samples=args.num_samples)
    print(f"  balanced, num_samples={args.num_samples:<10} -> {n2:>4} steps/epoch")
    print(f"\n  x max_epochs {args.max_epochs} = {n2 * args.max_epochs:,} total steps"
          f"   (upstream: 19,531)")

    expected = args.num_samples // args.batch_size
    if n2 != expected:
        print(f"\n  UNEXPECTED: got {n2}, expected {expected}. The sampler is not "
              "controlling epoch length.")
        sys.exit(1)
    print("\n  OK -- epoch length is driven by the sampler.")


if __name__ == "__main__":
    main()
