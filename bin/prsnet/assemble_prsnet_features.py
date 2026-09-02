"""
Gather per-chunk gene-PRS npz files into the artifacts the existing pipeline expects.

This is the seam. It emits exactly what plink_converter.py emits, so SPLIT_DATA,
TRAIN_KFOLD and EVALUATE_MODELS need no modification:

  genotype_data.h5   dataset 'genotypes' of shape (N, n_genes, n_thresholds),
                     attrs n_samples / n_snps / encoding='gene-prs' / encoded_shape
  phenotypes.csv     single column 'phenotype_1', aligned to the .fam sample order
  data_stats.json

The (N, G, 11) layout deliberately mirrors your existing 'one-hot' (N, M, 3) and
'two-dim' (N, M, 2) encodings, so GenotypeDataModule reports input_dim = n_genes
and n_channels = 11 without changes.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_phenotypes(fam_iids, phenotype_file=None, fam_path=None):
    """Align phenotypes to .fam order; mirrors plink_converter.py's coding rules."""
    if phenotype_file:
        raw = pd.read_csv(phenotype_file, sep=r"\s+", header=None, engine="python")
        if raw.shape[1] >= 3:
            raw.columns = ["FID", "IID", "phenotype"] + list(raw.columns[3:])
        elif raw.shape[1] == 2:
            raw.columns = ["IID", "phenotype"]
        else:
            raise ValueError("Phenotype file needs at least IID and a value column")
        # A header row would have parsed as strings; drop it if present.
        if not str(raw["IID"].iloc[0]).strip().lstrip("-").replace(".", "").isalnum():
            raw = raw.iloc[1:]
        raw["IID"] = raw["IID"].astype(str)
        lookup = dict(zip(raw["IID"], pd.to_numeric(raw["phenotype"], errors="coerce")))
        y = np.array([lookup.get(str(i), np.nan) for i in fam_iids], dtype=np.float64)
    else:
        fam = pd.read_csv(fam_path, sep=r"\s+", header=None,
                          names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"])
        y = pd.to_numeric(fam["PHENO"], errors="coerce").values.astype(np.float64)

    finite = y[~np.isnan(y)]
    if len(np.unique(finite)) == 2 and set(np.unique(finite)).issubset({1.0, 2.0}):
        y = y - 1.0
    y[y < 0] = np.nan
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True, help="chunk_*.npz files")
    ap.add_argument("--gene_order", required=True, help="canonical gene_order.txt")
    ap.add_argument("--fam", required=True, help="target_data.PH.fam")
    ap.add_argument("--phenotype_file", default=None)
    ap.add_argument("--output_h5", required=True)
    ap.add_argument("--output_pheno", required=True)
    ap.add_argument("--stats_file", required=True)
    ap.add_argument("--standardize", action="store_true",
                    help="Do NOT enable for benchmarking runs -- see note below")
    args = ap.parse_args()

    gene_order = [g.strip() for g in open(args.gene_order) if g.strip()]
    gene_pos = {g: i for i, g in enumerate(gene_order)}
    n_genes = len(gene_order)

    fam = pd.read_csv(args.fam, sep=r"\s+", header=None,
                      names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"], dtype={1: str})
    iids = fam["IID"].astype(str).values
    n_samples = len(iids)

    first = np.load(args.chunks[0], allow_pickle=False)
    n_thr = first["scores"].shape[2]
    thresholds = [str(t) for t in first["thresholds"]]

    features = np.zeros((n_samples, n_genes, n_thr), dtype=np.float32)
    seen = np.zeros(n_genes, dtype=bool)

    for path in sorted(args.chunks):
        blob = np.load(path, allow_pickle=False)
        chunk_iids = [str(s) for s in blob["iids"]]
        if chunk_iids != list(iids):
            raise ValueError(
                f"{path}: sample order does not match the .fam file. Every chunk must "
                "be scored against the same target_data.PH."
            )
        for j, gene in enumerate([str(g) for g in blob["genes"]]):
            k = gene_pos.get(gene)
            if k is None:
                raise ValueError(f"{path}: gene {gene!r} is not in gene_order.txt")
            features[:, k, :] = blob["scores"][:, j, :]
            seen[k] = True

    y = load_phenotypes(iids, args.phenotype_file, args.fam)

    if args.standardize:
        mu = features.mean(axis=0, keepdims=True)
        sd = features.std(axis=0, keepdims=True)
        features = (features - mu) / np.where(sd == 0, 1.0, sd)

    with h5py.File(args.output_h5, "w") as f:
        # Chunk = ONE sample. GenotypeDataset does random single-sample reads
        # (genotypes[actual_idx]); a multi-sample chunk would force a full
        # decompress of every neighbour on each __getitem__. At 19836x11 float32
        # a single-sample chunk is already ~0.85 MB, which is at the top of what
        # h5py's default 1 MB chunk cache will hold.
        f.create_dataset("genotypes", data=features, dtype="float32",
                         compression="gzip", compression_opts=4,
                         chunks=(1, n_genes, n_thr))
        f.create_dataset("sample_info/iid",
                         data=np.array([s.encode("utf-8") for s in iids]))
        f.create_dataset("snp_info/gene",
                         data=np.array([g.encode("utf-8") for g in gene_order]))
        f.attrs["n_samples"] = n_samples
        f.attrs["n_snps"] = n_genes          # 'n_snps' == gene axis, keeps the contract
        f.attrs["n_genes"] = n_genes
        f.attrs["n_thresholds"] = n_thr
        f.attrs["encoding"] = "gene-prs"
        f.attrs["encoded_shape"] = np.array(features.shape, dtype=np.int64)
        f.attrs["thresholds"] = np.array([t.encode("utf-8") for t in thresholds])

    pd.DataFrame({"phenotype_1": y}).to_csv(args.output_pheno, index=False)

    stats = {
        "n_samples": int(n_samples),
        "n_genes": int(n_genes),
        "n_thresholds": int(n_thr),
        "n_genes_scored": int(seen.sum()),
        "n_genes_all_zero": int((features.sum(axis=(0, 2)) == 0).sum()),
        "n_genes_missing_from_chunks": int((~seen).sum()),
        "encoding": "gene-prs",
        "thresholds": thresholds,
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "missing_phenotypes": int(np.isnan(y).sum()),
        "standardized": bool(args.standardize),
    }
    with open(args.stats_file, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(stats, indent=2))
    if args.standardize:
        print(
            "\nWARNING: --standardize centres and scales using ALL samples, including "
            "your test fold. That is exactly the train-only-preprocessing invariant you "
            "hold everywhere else. Prefer leaving this off and standardising inside the "
            "fold, in GenotypeDataModule."
        )


if __name__ == "__main__":
    main()