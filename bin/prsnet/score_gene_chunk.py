"""
Score one chunk of genes: PLINK clumping -> p-value thresholding -> per-gene PRS.

Runs inside a single Nextflow task so ~250 genes share one process slot instead of
spawning ~40,000 tiny tasks. Emits ONE npz per chunk rather than 11 .sscore files
per gene.

Faithful to upstream 2_data_generation.sh in the parts that matter:
  * clumping is done against the TARGET genotypes (target_data.PH), not the LD
    reference panel -- upstream builds ld_data.PH in step 1 and then never uses it
  * --clump takes TWO association files (the gene's own .assoc, then the full
    gwas.QC.txt) with --clump-index-first, so index SNPs come only from the gene
  * --clump-p1 1, --clump-r2 0.5, --clump-kb 250, --chr <gene's chromosome>
  * --score gwas.QC.txt 3 4 12 header   (3=SNP, 4=A1, 12=BETA)
  * 11 p-value thresholds -> d_input=11

Deviating from upstream deliberately in one place: upstream passes the multi-column
`.assoc` and `.clumped` files directly to `--extract`, whose first column is CHR,
not a variant ID. This script builds real one-ID-per-line lists instead. See
PRSNET_INTEGRATION.md.

Output npz
----------
  scores      float32 [n_samples, n_genes_in_chunk, n_thresholds]
  genes       unicode [n_genes_in_chunk]
  iids        unicode [n_samples]      (target .fam order)
  thresholds  unicode [n_thresholds]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = ["1e-5", "1e-4", "1e-3", "0.01", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "1"]


def write_range_list(path: str, thresholds) -> None:
    """
    Reconstruction of upstream's `range_list_full`, which is referenced by
    --q-score-range but absent from the repository. Nested ranges from 0, one per
    threshold; the range names become the .sscore filename suffixes that
    generate_prsnet_data.py expects.
    """
    with open(path, "w") as fh:
        for t in thresholds:
            fh.write(f"{t} 0 {t}\n")


def read_fam(bfile: str):
    fam = pd.read_csv(f"{bfile}.fam", sep=r"\s+", header=None,
                      names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"], dtype={0: str, 1: str})
    return fam["IID"].astype(str).values


def run(cmd, log):
    res = subprocess.run(cmd, capture_output=True, text=True)
    log.write(f"$ {' '.join(cmd)}\n{res.stdout}\n{res.stderr}\n")
    return res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True, help="chunk_XXXX.tsv: gene<TAB>chrom<TAB>snps")
    ap.add_argument("--bfile", required=True, help="target_data.PH prefix")
    ap.add_argument("--gwas_qc", required=True)
    ap.add_argument("--snp_pvalue", required=True, help="two columns: SNP P")
    ap.add_argument("--freq_file", default=None,
                    help="plink2 --read-freq file computed on TRAINING samples only")
    ap.add_argument("--clump_r2", type=float, default=0.5)
    ap.add_argument("--clump_kb", type=int, default=250)
    ap.add_argument("--clump_p1", type=float, default=1.0)
    ap.add_argument("--score_cols", default="3 4 12",
                    help="SNP, A1, BETA column numbers in gwas.QC.txt")
    ap.add_argument("--plink_bin", default="plink",
                    help="plink 1.9 executable (on PATH by default)")
    ap.add_argument("--plink2_bin", default="plink2",
                    help="plink2 executable. Give a full path if plink2 is a "
                         "repo-local binary rather than on PATH.")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", type=int, default=8000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", default="chunk.log")
    args = ap.parse_args()

    for label, exe in (("plink", args.plink_bin), ("plink2", args.plink2_bin)):
        if shutil.which(exe) is None and not os.access(exe, os.X_OK):
            sys.exit(f"FATAL: {label} executable not found or not executable: {exe!r}")

    genes, gene_chrom, gene_snps = [], {}, {}
    with open(args.chunk) as fh:
        for line in fh:
            gene, chrom, snps = line.rstrip("\n").split("\t")
            genes.append(gene)
            gene_chrom[gene] = chrom
            gene_snps[gene] = snps.split(",")

    # Index gwas.QC.txt once per chunk so per-gene .assoc files are cheap to cut.
    gwas = pd.read_csv(args.gwas_qc, sep="\t", dtype=str)
    gwas_snps = gwas["SNP"].values
    gwas_by_snp = {s: i for i, s in enumerate(gwas_snps)}

    iids = read_fam(args.bfile)
    n_samples, n_thr = len(iids), len(THRESHOLDS)
    scores = np.zeros((n_samples, len(genes), n_thr), dtype=np.float32)
    iid_pos = {s: i for i, s in enumerate(iids)}

    n_failed_clump, n_missing_sscore = 0, 0

    with tempfile.TemporaryDirectory() as tmp, open(args.log, "w") as log:
        range_list = os.path.join(tmp, "range_list_full")
        write_range_list(range_list, THRESHOLDS)

        for gi, gene in enumerate(genes):
            rows = [gwas_by_snp[s] for s in gene_snps[gene] if s in gwas_by_snp]
            if not rows:
                n_failed_clump += 1
                continue

            # Per-gene association file (upstream's gene_gwas/*.assoc), space-separated.
            gene_assoc = os.path.join(tmp, "gene.assoc")
            gwas.iloc[rows].to_csv(gene_assoc, sep=" ", index=False)

            # Real variant-ID list for --extract (upstream passes the .assoc itself).
            snp_file = os.path.join(tmp, "gene.snps")
            with open(snp_file, "w") as fh:
                fh.write("\n".join(gwas_snps[rows]) + "\n")

            clump_out = os.path.join(tmp, "gene")
            rc = run([
                args.plink_bin, "--bfile", args.bfile,
                "--extract", snp_file,
                "--chr", gene_chrom[gene],
                "--clump-p1", str(args.clump_p1),
                "--clump-r2", str(args.clump_r2),
                "--clump-kb", str(args.clump_kb),
                "--clump", gene_assoc, args.gwas_qc,
                "--clump-index-first",
                "--clump-snp-field", "SNP", "--clump-field", "P",
                "--memory", str(args.memory), "--threads", str(args.threads),
                "--out", clump_out,
            ], log)

            clumped = f"{clump_out}.clumped"
            if rc != 0 or not os.path.exists(clumped):
                # Genuinely common: small genes where nothing survives clumping.
                # Leave the row at zero rather than dropping the gene -- the feature
                # axis must stay aligned with the GGI node axis.
                n_failed_clump += 1
                continue

            index_snps = os.path.join(tmp, "gene.index.snps")
            n_index = 0
            with open(clumped) as fh, open(index_snps, "w") as out:
                fh.readline()                       # header: CHR F SNP BP P TOTAL ...
                for line in fh:
                    f = line.split()
                    if len(f) > 2:
                        out.write(f[2] + "\n")      # column 3 = index SNP id
                        n_index += 1
            if n_index == 0:
                n_failed_clump += 1
                continue

            score_out = os.path.join(tmp, "gene.score")
            cmd = [
                args.plink2_bin, "--bfile", args.bfile,
                "--chr", gene_chrom[gene],
                "--score", args.gwas_qc, *args.score_cols.split(), "header",
                "--q-score-range", range_list, args.snp_pvalue,
                "--extract", index_snps,
                "--memory", str(args.memory), "--threads", str(args.threads),
                "--out", score_out,
            ]
            if args.freq_file:
                cmd += ["--read-freq", args.freq_file]
            run(cmd, log)

            for ti, thr in enumerate(THRESHOLDS):
                path = f"{score_out}.{thr}.sscore"
                if not os.path.exists(path):
                    n_missing_sscore += 1
                    continue
                df = pd.read_csv(path, sep=r"\s+")
                col = "SCORE1_AVG" if "SCORE1_AVG" in df.columns else df.columns[-1]
                id_col = "IID" if "IID" in df.columns else df.columns[1]
                vals = np.nan_to_num(df[col].values.astype(np.float32))
                order = np.array([iid_pos.get(str(s), -1) for s in df[id_col].values])
                keep = order >= 0
                scores[order[keep], gi, ti] = vals[keep]

    np.savez_compressed(
        args.output,
        scores=scores,
        genes=np.array(genes, dtype=object).astype("U"),
        iids=np.array(iids, dtype=object).astype("U"),
        thresholds=np.array(THRESHOLDS, dtype=object).astype("U"),
    )
    print(json.dumps({
        "chunk": Path(args.chunk).name,
        "n_genes": len(genes),
        "n_samples": n_samples,
        "n_genes_no_clump": n_failed_clump,
        "n_missing_sscore": n_missing_sscore,
    }, indent=2))


if __name__ == "__main__":
    main()