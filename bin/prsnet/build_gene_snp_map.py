"""
Map QC'd GWAS SNPs to genes and emit deterministic gene chunks for scatter.

Upstream runs one `bedtools intersect` per gene (19,831 subprocess launches under
GNU parallel, repeated for every chromosome). This does the same thing in a single
sorted intersect against one concatenated interval file, which is ~2 orders of
magnitude faster and, more importantly, produces a *deterministic* gene ordering
instead of relying on os.listdir().

Outputs
-------
  gene_order.txt                 canonical gene order (one symbol per line)
  chunks/chunk_XXXX.tsv          gene<TAB>chrom<TAB>comma-separated SNP ids
  gene_snp_map_stats.json

The gene BED intervals ship as GRCh37/hg19 with the +/-10kb flank ALREADY applied
(verified against APOE, BRCA1, BRCA2, CFTR, HBB), and are 'chr'-prefixed. Do not
add your own padding. Your target .bim must also be GRCh37 -- UKB imputed data is,
so no liftover is needed, but check before you trust the intersect.
"""

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def canonical_gene_order(gene_bed_dir: str):
    """Sorted by chromosome number, then gene symbol. Filesystem-independent."""
    genes = []
    for chrom in range(1, 23):
        d = Path(gene_bed_dir) / f"chr{chrom}"
        if not d.is_dir():
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".bed"):
                genes.append((chrom, fn[:-4], d / fn))
    if not genes:
        raise SystemExit(
            f"FATAL: no .bed files found under {gene_bed_dir!r}.\n"
            "Expected layout:  <gene_bed_dir>/chr1/<GENE>.bed ... <gene_bed_dir>/chr22/<GENE>.bed\n"
            "In the upstream repo this is the 'gene_bed_files_10kb' directory."
        )
    return genes



def concat_gene_beds(genes, out_path: str) -> None:
    """One BED with the gene symbol in column 4."""
    with open(out_path, "w") as out:
        for chrom, symbol, path in genes:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith(("#", "track", "browser")):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    out.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{symbol}\n")


def gwas_to_bed(gwas_qc: str, out_path: str) -> int:
    """gwas.QC.txt (CHR BP SNP A1 A2 N SE P OR INFO MAF BETA) -> BED4."""
    n = 0
    with open(gwas_qc) as fh, open(out_path, "w") as out:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for col in ("CHR", "BP", "SNP"):
            if col not in idx:
                raise ValueError(f"gwas.QC.txt is missing column {col}; got {header}")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            bp = f[idx["BP"]]
            if bp in ("NA", ""):
                continue
            pos = int(float(bp))
            chrom = f[idx["CHR"]]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            out.write(f"{chrom}\t{pos - 1}\t{pos}\t{f[idx['SNP']]}\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwas_qc", required=True)
    ap.add_argument("--gene_bed_dir", required=True)
    ap.add_argument("--genes_per_chunk", type=int, default=250)
    ap.add_argument("--min_snps_per_gene", type=int, default=1)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "chunks").mkdir(parents=True, exist_ok=True)

    genes = canonical_gene_order(args.gene_bed_dir)
    gene_names = [g[1] for g in genes]

    with open(outdir / "gene_order.txt", "w") as fh:
        fh.write("\n".join(gene_names) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        genes_bed = os.path.join(tmp, "genes.bed")
        snps_bed = os.path.join(tmp, "gwas.bed")
        concat_gene_beds(genes, genes_bed)
        n_snps = gwas_to_bed(args.gwas_qc, snps_bed)

        hits = os.path.join(tmp, "hits.tsv")
        with open(hits, "w") as out:
            subprocess.run(
                ["bedtools", "intersect", "-a", snps_bed, "-b", genes_bed, "-wa", "-wb"],
                stdout=out, check=True,
            )

        gene_to_snps = defaultdict(list)
        gene_chrom = {}
        seen = set()
        with open(hits) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                snp, chrom, gene = f[3], f[0], f[7]
                if (gene, snp) in seen:
                    continue
                seen.add((gene, snp))
                gene_to_snps[gene].append(snp)
                gene_chrom[gene] = chrom.replace("chr", "")

    kept = [g for g in gene_names if len(gene_to_snps.get(g, [])) >= args.min_snps_per_gene]

    n_chunks = 0
    for start in range(0, len(kept), args.genes_per_chunk):
        block = kept[start:start + args.genes_per_chunk]
        path = outdir / "chunks" / f"chunk_{n_chunks:04d}.tsv"
        with open(path, "w") as fh:
            for gene in block:
                fh.write(f"{gene}\t{gene_chrom[gene]}\t{','.join(gene_to_snps[gene])}\n")
        n_chunks += 1

    stats = {
        "n_genes_total": len(gene_names),
        "n_genes_with_snps": len(kept),
        "n_genes_empty": len(gene_names) - len(kept),
        "n_gwas_snps": n_snps,
        "n_snp_gene_pairs": sum(len(v) for v in gene_to_snps.values()),
        "n_chunks": n_chunks,
        "genes_per_chunk": args.genes_per_chunk,
    }
    with open(outdir / "gene_snp_map_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(stats, indent=2))
    print(
        "\nNOTE: genes with no mapped SNPs are still nodes in the GGI graph. They are "
        "carried as all-zero feature rows downstream so the feature axis stays aligned "
        "with the graph node axis. Do not drop them."
    )


if __name__ == "__main__":
    main()