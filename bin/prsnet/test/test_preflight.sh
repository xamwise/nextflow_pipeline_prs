#!/usr/bin/env bash
# PRS-Net preflight. Runs in seconds, touches no compute.
#
# Every check here corresponds to something that fails SILENTLY downstream --
# an empty awk output, a merge that matches nothing, a chunk of all-zero scores.
# Run this before the pipeline, not after it disappoints you.
#
#   bash tests/prsnet/test_preflight.sh \
#       --base_dir     /path/to/nextflow_pipeline_prs \
#       --base_gwas    data/gwas/CRC_ukbfree.tsv \
#       --target       data/qc/UKB_CRC.QC \
#       --ld_ref       data/ld/ld_panel

set -uo pipefail

BASE_DIR="."; BASE_GWAS=""; TARGET=""; LD_REF=""
PRSNET_DIR="data/supplement_data/prsnet"
BED_SUBDIR="gene_bed_files_10kb"

while [[ $# -gt 0 ]]; do
  case $1 in
    --base_dir)    BASE_DIR="$2"; shift 2 ;;
    --base_gwas)   BASE_GWAS="$2"; shift 2 ;;
    --target)      TARGET="$2"; shift 2 ;;
    --ld_ref)      LD_REF="$2"; shift 2 ;;
    --prsnet_dir)  PRSNET_DIR="$2"; shift 2 ;;
    --bed_subdir)  BED_SUBDIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$BASE_DIR" || { echo "cannot cd to $BASE_DIR"; exit 2; }
PASS=0; FAIL=0; WARN=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }

echo
echo "PRS-Net preflight  ($(pwd))"
echo "=============================================================="

# ---------------------------------------------------------------- 1
echo
echo "1. Executables"
for exe in plink bedtools Rscript python; do
  if command -v "$exe" >/dev/null 2>&1; then ok "$exe -> $(command -v $exe)"
  else bad "$exe not on PATH"; fi
done
if [[ -x bin/plink2 ]]; then ok "plink2 -> $(pwd)/bin/plink2"
elif command -v plink2 >/dev/null 2>&1; then warn "plink2 on PATH but not at bin/plink2 -- set params.prsnet.plink2_bin accordingly"
else bad "plink2 not found at bin/plink2 or on PATH"; fi

# plink 1.9 vs 2.x: --clump only exists in 1.9
if command -v plink >/dev/null 2>&1; then
  if plink --version 2>&1 | head -1 | grep -qi "v1\.9\|PLINK v1"; then ok "plink is 1.9 (--clump requires it)"
  else warn "plink may not be 1.9: $(plink --version 2>&1 | head -1). --clump does not exist in plink2."; fi
fi

# ---------------------------------------------------------------- 2
echo
echo "2. Python packages"
python - <<'PY'
import importlib.util, sys
need = ["numpy", "pandas", "h5py", "torch", "sklearn", "yaml"]
miss = [m for m in need if importlib.util.find_spec(m) is None]
for m in need:
    print(("  [PASS] " if m not in miss else "  [FAIL] ") + f"import {m}")
sys.exit(1 if miss else 0)
PY
if [[ $? -eq 0 ]]; then PASS=$((PASS+6)); else FAIL=$((FAIL+1)); fi

# ---------------------------------------------------------------- 3
echo
echo "3. PRS-Net assets  ($PRSNET_DIR)"
BEDDIR="$PRSNET_DIR/$BED_SUBDIR"
if [[ -d "$BEDDIR/chr1" ]]; then
  NBED=$(find "$BEDDIR" -name '*.bed' 2>/dev/null | wc -l)
  NCHR=$(find "$BEDDIR" -maxdepth 1 -type d -name 'chr*' | wc -l)
  ok "gene BEDs: $NBED files across $NCHR chromosome dirs"
  [[ "$NCHR" -eq 22 ]] || warn "expected 22 chromosome dirs, found $NCHR"
  # build check: APOE should be hg19 (chr19:45,399,011-45,422,650 with +/-10kb)
  APOE=$(cat "$BEDDIR/chr19/APOE.bed" 2>/dev/null | head -1 | awk '{print $2}')
  if [[ -n "$APOE" ]]; then
    if [[ "$APOE" -gt 45000000 && "$APOE" -lt 46000000 ]]; then ok "gene BEDs are GRCh37/hg19 (APOE start $APOE)"
    else warn "APOE start is $APOE -- expected ~45,399,011 for hg19. Check the build."; fi
  fi
else
  bad "no chr1/ under $BEDDIR -- adjust --prsnet_dir/--bed_subdir"
fi
[[ -f "$PRSNET_DIR/ggi_graph.bin" ]] && ok "ggi_graph.bin present" || bad "ggi_graph.bin missing from $PRSNET_DIR"
[[ -f bin/prsnet/mismatch.R ]] && ok "mismatch.R present" || bad "bin/prsnet/mismatch.R missing (copy verbatim from upstream)"
for f in prepare_base_gwas.py build_gene_snp_map.py score_gene_chunk.py \
         assemble_prsnet_features.py convert_ggi_graph.py; do
  [[ -f "bin/prsnet/$f" ]] && ok "bin/prsnet/$f" || bad "bin/prsnet/$f missing"
done
[[ -f models/prsnet_model.py ]] && ok "models/prsnet_model.py" || bad "models/prsnet_model.py missing"

# ---------------------------------------------------------------- 4
if [[ -n "$BASE_GWAS" ]]; then
echo
echo "4. Base GWAS  ($BASE_GWAS)"
if [[ ! -f "$BASE_GWAS" ]]; then bad "file not found"; else
  HDR=$(head -1 "$BASE_GWAS")
  EXPECT=$'CHR\tBP\tSNP\tA1\tA2\tN\tSE\tP\tOR\tINFO\tMAF'
  if [[ "$HDR" == "$EXPECT"* ]]; then ok "column order matches CHR BP SNP A1 A2 N SE P OR INFO MAF"
  else bad "header is: $HDR"; fi
  NCOL=$(head -1 "$BASE_GWAS" | awk -F'\t' '{print NF}')
  if   [[ "$NCOL" -eq 12 ]]; then ok "12 columns -- BETA already appended"
  elif [[ "$NCOL" -eq 11 ]]; then warn "11 columns -- run bin/prsnet/prepare_base_gwas.py to append BETA"
  else bad "$NCOL columns (expected 11 or 12); is it really tab-separated?"; fi

  awk -F'\t' 'NR>1 && NR<=200001 {
      if ($9+0>0) {n++; s+=$9}
      if ($1 ~ /^chr/) pfx=1
      if ($10=="" || $10=="NA") noinfo++
      if ($11=="" || $11=="NA") nomaf++
    } END {
      if (n>0) printf "  [INFO] mean OR over first 200k rows: %.4f\n", s/n
      if (pfx) print "  [WARN] CHR carries a chr prefix -- use --strip_chr_prefix"
      if (noinfo>0) printf "  [WARN] %d rows with empty/NA INFO\n", noinfo
      if (nomaf>0)  printf "  [WARN] %d rows with empty/NA MAF\n", nomaf
    }' "$BASE_GWAS"
  echo "  (mean OR near 1.0 is expected; near 0.0 means the column holds BETA already)"
fi
fi

# ---------------------------------------------------------------- 5
if [[ -n "$TARGET" ]]; then
echo
echo "5. Target genotypes  ($TARGET)"
ALL=1; for e in bed bim fam; do [[ -f "$TARGET.$e" ]] || { bad "$TARGET.$e missing"; ALL=0; }; done
if [[ $ALL -eq 1 ]]; then
  ok "bed/bim/fam present ($(wc -l < "$TARGET.fam") samples, $(wc -l < "$TARGET.bim") variants)"
  DUP=$(cut -f1,4 "$TARGET.bim" | sort | uniq -d | wc -l)
  if [[ "$DUP" -eq 0 ]]; then ok "no duplicate (CHR, BP) pairs"
  else warn "$DUP duplicate (CHR, BP) positions -- the harmonisation awk keys on these and keeps the last; multi-allelic sites will collide"; fi
  if head -1 "$TARGET.bim" | cut -f1 | grep -qi '^chr'; then
    warn "target .bim CHR carries a chr prefix -- must match the base GWAS convention"
  else ok "target .bim CHR is bare numeric"; fi
  echo "  [INFO] example variant ID: $(head -1 "$TARGET.bim" | cut -f2)"
fi
fi

# ---------------------------------------------------------------- 6
if [[ -n "$LD_REF" ]]; then
echo
echo "6. LD panel  ($LD_REF)"
ALL=1; for e in bed bim fam; do [[ -f "$LD_REF.$e" ]] || { bad "$LD_REF.$e missing"; ALL=0; }; done
if [[ $ALL -eq 1 ]]; then
  ok "bed/bim/fam present ($(wc -l < "$LD_REF.fam") samples, $(wc -l < "$LD_REF.bim") variants)"
  if [[ -n "$TARGET" && -f "$TARGET.bim" ]]; then
    SHARED=$(awk 'NR==FNR{a[$2];next} $2 in a' "$TARGET.bim" "$LD_REF.bim" | wc -l)
    TOT=$(wc -l < "$LD_REF.bim")
    PCT=$(awk -v s="$SHARED" -v t="$TOT" 'BEGIN{printf "%.1f", t?100*s/t:0}')
    if   (( $(echo "$PCT > 50" | bc -l) )); then ok "variant IDs match target: $SHARED shared (${PCT}% of panel)"
    elif (( $(echo "$PCT > 5"  | bc -l) )); then warn "only ${PCT}% of panel IDs found in target -- check ID conventions"
    else bad "only ${PCT}% of panel IDs found in target. mismatch.R merges on (SNP, CHR, BP) jointly and will match almost nothing."; fi
  fi
  # overlap with target samples would defeat the point
  if [[ -n "$TARGET" && -f "$TARGET.fam" ]]; then
    OV=$(awk 'NR==FNR{a[$2];next} $2 in a' "$TARGET.fam" "$LD_REF.fam" | wc -l)
    if [[ "$OV" -eq 0 ]]; then ok "no sample overlap with target"
    else warn "$OV samples appear in BOTH the LD panel and the target"; fi
  fi
fi
fi

# ---------------------------------------------------------------- 7
echo
echo "7. Graph / gene-count alignment  (KNOWN BLOCKER)"
if [[ -d "$BEDDIR/chr1" ]]; then
  NBED=$(find "$BEDDIR" -name '*.bed' | wc -l)
  echo "  [INFO] gene BED files : $NBED"
  echo "  [INFO] ggi_graph nodes: 19836 (measured by parsing the DGL binary)"
  if [[ "$NBED" -eq 19836 ]]; then ok "counts agree"
  else bad "counts differ by $((19836-NBED)). The feature axis and graph node axis cannot be aligned from the shipped artifacts -- resolve before trusting any result."; fi
fi

echo
echo "=============================================================="
echo "$PASS passed, $FAIL failed, $WARN warnings"
[[ $FAIL -eq 0 ]] || echo "Fix failures before running the pipeline."
exit $(( FAIL > 0 ? 1 : 0 ))
