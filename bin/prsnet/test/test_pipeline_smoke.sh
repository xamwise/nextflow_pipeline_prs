#!/usr/bin/env bash
# PRS-Net pipeline smoke test on ONE chromosome.
#
# Runs the real feature-construction path end to end on chr22 (the smallest
# autosome, ~450 genes) and asserts at every stage. Purpose is threefold:
#   1. prove the stages actually chain together on your data
#   2. measure wall time and gzip ratio so you can extrapolate the full run
#   3. catch silent-empty failures before they cost you a full scatter
#
#   bash tests/prsnet/test_pipeline_smoke.sh \
#       --base_dir  /path/to/nextflow_pipeline_prs \
#       --base_gwas data/gwas/CRC_ukbfree.prepared.tsv \
#       --target    out/prsnet/base_qc/target_data.PH \
#       --chrom     22
#
# Expects PRSNET_BASE_QC to have already produced target_data.PH, gwas.QC.txt
# and SNP.pvalue. Run stage 7 of the preflight first.

set -uo pipefail

BASE_DIR="."; BASE_GWAS=""; TARGET=""; CHROM="22"
PRSNET_DIR="data/supplement_data/prsnet"; BED_SUBDIR="gene_bed_files_10kb"
OUTDIR="out/prsnet/smoke"; GWAS_QC=""; SNP_PVALUE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --base_dir)   BASE_DIR="$2"; shift 2 ;;
    --base_gwas)  BASE_GWAS="$2"; shift 2 ;;
    --target)     TARGET="$2"; shift 2 ;;
    --chrom)      CHROM="$2"; shift 2 ;;
    --gwas_qc)    GWAS_QC="$2"; shift 2 ;;
    --snp_pvalue) SNP_PVALUE="$2"; shift 2 ;;
    --outdir)     OUTDIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$BASE_DIR" || exit 2
mkdir -p "$OUTDIR"
GWAS_QC="${GWAS_QC:-$(dirname "$TARGET")/gwas.QC.txt}"
SNP_PVALUE="${SNP_PVALUE:-$(dirname "$TARGET")/SNP.pvalue}"
PLINK2="${PWD}/bin/plink2"
BEDDIR="$PRSNET_DIR/$BED_SUBDIR"

die() { echo "  [FAIL] $1"; exit 1; }
ok()  { echo "  [PASS] $1"; }

echo
echo "PRS-Net pipeline smoke test -- chr${CHROM} only"
echo "=============================================================="

# ---------------------------------------------------------------- 1
echo
echo "1. Inputs"
for f in "$GWAS_QC" "$SNP_PVALUE" "$TARGET.bed"; do
  [[ -f "$f" ]] || die "missing $f  (run PRSNET_BASE_QC first)"
done
NQC=$(( $(wc -l < "$GWAS_QC") - 1 ))
[[ "$NQC" -gt 1000 ]] || die "gwas.QC.txt has only $NQC rows -- base QC silently emptied it. \
Check the MAF/INFO awk filters and the target/LD variant-ID conventions."
ok "gwas.QC.txt: $NQC variants"
ok "target: $(wc -l < "$TARGET.fam") samples, $(wc -l < "$TARGET.bim") variants"

# ---------------------------------------------------------------- 2
echo
echo "2. Gene-SNP mapping (chr${CHROM})"
SMOKE_BED="$OUTDIR/bed_chr${CHROM}"
rm -rf "$SMOKE_BED"; mkdir -p "$SMOKE_BED"
cp -r "$BEDDIR/chr${CHROM}" "$SMOKE_BED/" || die "no $BEDDIR/chr${CHROM}"
NGENE_IN=$(find "$SMOKE_BED" -name '*.bed' | wc -l)
echo "  genes on chr${CHROM}: $NGENE_IN"

T0=$(date +%s)
python bin/prsnet/build_gene_snp_map.py \
    --gwas_qc "$GWAS_QC" \
    --gene_bed_dir "$SMOKE_BED" \
    --genes_per_chunk 250 \
    --outdir "$OUTDIR" > "$OUTDIR/map.log" 2>&1 || { cat "$OUTDIR/map.log"; die "mapping failed"; }
T_MAP=$(( $(date +%s) - T0 ))

NMAPPED=$(python -c "import json;print(json.load(open('$OUTDIR/gene_snp_map_stats.json'))['n_genes_with_snps'])")
NCHUNK=$(ls "$OUTDIR"/chunks/chunk_*.tsv 2>/dev/null | wc -l)
[[ "$NMAPPED" -gt 0 ]] || die "no genes received any SNP. Almost always a build or \
chr-prefix mismatch between the GWAS and the gene BEDs (BEDs are hg19, chr-prefixed)."
ok "mapped $NMAPPED/$NGENE_IN genes into $NCHUNK chunk(s) in ${T_MAP}s"
awk '{n+=split($3,a,",")} END {printf "  mean SNPs per gene: %.1f\n", n/NR}' "$OUTDIR"/chunks/chunk_0000.tsv

# ---------------------------------------------------------------- 3
echo
echo "3. Scoring one chunk"
T0=$(date +%s)
python bin/prsnet/score_gene_chunk.py \
    --chunk "$OUTDIR/chunks/chunk_0000.tsv" \
    --bfile "$TARGET" \
    --gwas_qc "$GWAS_QC" \
    --snp_pvalue "$SNP_PVALUE" \
    --plink_bin plink --plink2_bin "$PLINK2" \
    --threads 4 --memory 8000 \
    --output "$OUTDIR/chunk_0000.npz" \
    --log "$OUTDIR/chunk_0000.log" > "$OUTDIR/score.log" 2>&1 \
    || { tail -30 "$OUTDIR/chunk_0000.log"; die "scoring failed"; }
T_CHUNK=$(( $(date +%s) - T0 ))
cat "$OUTDIR/score.log"

python - "$OUTDIR/chunk_0000.npz" <<'PY' || exit 1
import sys, numpy as np
z = np.load(sys.argv[1]); s = z["scores"]
nz = (s.sum(axis=(0, 2)) != 0).sum()
print(f"  scores {s.shape}, {nz}/{s.shape[1]} genes non-zero, "
      f"finite={np.isfinite(s).all()}")
if nz == 0:
    print("  [FAIL] every gene scored zero. Usual causes: plink2 path wrong, "
          "--extract matched nothing, or BETA column empty.")
    sys.exit(1)
if nz < 0.2 * s.shape[1]:
    print(f"  [WARN] only {100*nz/s.shape[1]:.0f}% of genes scored. Expected for "
          "small genes where nothing survives clumping, but check chunk_0000.log "
          "if it looks too low.")
print("  [PASS] chunk scored")
PY
echo "  wall time: ${T_CHUNK}s for this chunk"

# ---------------------------------------------------------------- 4
echo
echo "4. Assembly"
python bin/prsnet/assemble_prsnet_features.py \
    --chunks "$OUTDIR"/chunk_*.npz \
    --gene_order "$OUTDIR/gene_order.txt" \
    --fam "$TARGET.fam" \
    --phenotype_file data/raw/UKB_ALZ/UKB_ALZ.pheno \
    --output_h5 "$OUTDIR/genotype_data.h5" \
    --output_pheno "$OUTDIR/phenotypes.csv" \
    --stats_file "$OUTDIR/data_stats.json" > "$OUTDIR/assemble.log" 2>&1 \
    || { cat "$OUTDIR/assemble.log"; die "assembly failed"; }

python - "$OUTDIR/genotype_data.h5" "$OUTDIR/phenotypes.csv" <<'PY' || exit 1
import sys, h5py, pandas as pd
with h5py.File(sys.argv[1]) as f:
    d = f["genotypes"]
    print(f"  h5 shape {d.shape}, chunks {d.chunks}, encoding={f.attrs['encoding']}")
    assert d.ndim == 3 and d.shape[2] == 11, "expected (N, n_genes, 11)"
    assert d.chunks[0] == 1, "chunk spans >1 sample; random reads will thrash"
n = len(pd.read_csv(sys.argv[2]))
print(f"  phenotypes: {n} rows")
print("  [PASS] artifacts match the CONVERT_PLINK contract")
PY

RAW=$(python -c "
import h5py
f = h5py.File('$OUTDIR/genotype_data.h5'); d = f['genotypes']
print(d.shape[0]*d.shape[1]*d.shape[2]*4)")
DISK=$(wc -c < "$OUTDIR/genotype_data.h5" | tr -d ' ')
python - "$RAW" "$DISK" <<'PY'
import sys
raw, disk = int(sys.argv[1]), int(sys.argv[2])
print(f"  gzip ratio: {raw/disk:.1f}x  "
      f"(raw {raw/1048576:.1f} MB -> on disk {disk/1048576:.1f} MB)")
PY

# ---------------------------------------------------------------- 5
echo
echo "5. Full-run extrapolation"
python - "${NGENE_IN:-0}" "${T_CHUNK:-0}" "${NMAPPED:-0}" "${RAW:-0}" "${DISK:-1}" <<'PY'
import sys
vals = [int(float(x)) if str(x).strip() else 0 for x in sys.argv[1:6]]
ngene_chr, t_chunk, nmapped, raw, disk = vals
disk = max(disk, 1)
TOTAL_GENES = 19831
chunks = -(-TOTAL_GENES // 250)
genes_in_chunk = min(250, nmapped)
per_gene = t_chunk / max(genes_in_chunk, 1)
print(f"  {TOTAL_GENES} genes -> {chunks} chunks at 250/chunk")
print(f"  ~{per_gene:.2f}s per gene  ->  ~{chunks*250*per_gene/3600:.1f} core-hours per phenotype")
for j in (10, 40, 80):
    print(f"     {j:>2} concurrent tasks: ~{chunks*250*per_gene/3600/j:.1f} h wall")
ratio = raw / disk
scale = TOTAL_GENES / max(nmapped, 1)
print(f"  storage: gzip {ratio:.1f}x -> full feature set ~{raw*scale/ratio/1024**3:.1f} GB on disk")
PY

echo
echo "=============================================================="
echo "Smoke test complete. Artifacts in $OUTDIR"
echo "Next: point tests/prsnet/test_integration.py at this h5 to confirm the"
echo "      data module and model consume it, then scale to all chromosomes."