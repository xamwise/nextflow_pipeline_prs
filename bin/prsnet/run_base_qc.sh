#!/usr/bin/env bash
# PRS-Net base QC -- standalone. Reimplements upstream 1_data_preprocess.sh,
# which cannot run as committed (it uses `VAR = $1` with spaces, so every
# variable expands empty).
#
# Runnable outside Nextflow so you can pilot without standing up the workflow.
# PRSNET_BASE_QC calls this same script, so there is one implementation.
#
#   bash bin/prsnet/run_base_qc.sh \
#       --base_gwas data/supplement_data/sum_stats/alzheimers_sumstats_hg37.QC \
#       --target    data/raw/UKB_ALZ/UKB_ALZ \
#       --ld_ref    data/raw/UKB_ALZ/UKB_ALZ \
#       --pheno     data/raw/UKB_ALZ/UKB_ALZ.pheno \
#       --outdir    out/prsnet/base_qc
#
# Written for bash 3.2 (macOS default) -- no associative arrays, no ${var,,}.
#
# ---------------------------------------------------------------------------
# NOTE ON THE `NR==FNR` IDIOM
# ---------------------------------------------------------------------------
# The standard two-file awk join `NR==FNR {lookup; next} {filter}` is BROKEN when
# the first file is empty: FNR resets per file, so with zero records in file 1,
# NR==FNR remains true for every record of file 2 and the entire second file is
# consumed by the lookup branch, producing empty output.
#
# This bites in practice -- e.g. when --test-missing yields no variants to drop,
# which is the normal case for a clean cohort. Every join below is therefore
# guarded either by `FILENAME==ARGV[1]` or by a shell-level emptiness check.
# ---------------------------------------------------------------------------

set -uo pipefail

# --- locate our own helpers --------------------------------------------------
# Standalone runs have CWD = repo root; Nextflow runs have CWD = a work dir. So
# resolve siblings from $0 rather than assuming either. --base_dir overrides.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

BASE_GWAS=""; TARGET=""; LD_REF=""; PHENO=""; OUTDIR="out/prsnet/base_qc"
MIN_MAF=0.001; MIN_INFO=0.3
T_MAF=0.01; T_HWE=1e-6; T_GENO=0.01; T_MIND=0.01
MEM=16000; PLINK="plink"; PLINK2="./bin/plink2"
EFFECT_ALLELE="A1"; STRIP_CHR=""; SKIP_PREPARE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --base_gwas) BASE_GWAS="$2"; shift 2 ;;
    --target)    TARGET="$2";    shift 2 ;;
    --ld_ref)    LD_REF="$2";    shift 2 ;;
    --pheno)     PHENO="$2";     shift 2 ;;
    --outdir)    OUTDIR="$2";    shift 2 ;;
    --min_maf)   MIN_MAF="$2";   shift 2 ;;
    --min_info)  MIN_INFO="$2";  shift 2 ;;
    --target_maf)  T_MAF="$2";   shift 2 ;;
    --target_hwe)  T_HWE="$2";   shift 2 ;;
    --target_geno) T_GENO="$2";  shift 2 ;;
    --target_mind) T_MIND="$2";  shift 2 ;;
    --memory)    MEM="$2";       shift 2 ;;
    --plink)     PLINK="$2";     shift 2 ;;
    --plink2)    PLINK2="$2";    shift 2 ;;
    --effect_allele) EFFECT_ALLELE="$2"; shift 2 ;;
    --base_dir)      SCRIPT_DIR="$2/bin/prsnet"; shift 2 ;;
    --strip_chr_prefix) STRIP_CHR="--strip_chr_prefix"; shift ;;
    --skip_prepare)     SKIP_PREPARE=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

for helper in prepare_base_gwas.py mismatch.R; do
  [ -f "$SCRIPT_DIR/$helper" ] || {
    echo "FATAL: $SCRIPT_DIR/$helper not found." >&2
    echo "       Expected it beside this script. Pass --base_dir <repo root> if" >&2
    echo "       run_base_qc.sh lives outside <repo>/bin/prsnet/." >&2
    exit 2; }
done

[ -n "$BASE_GWAS" ] || { echo "--base_gwas required"; exit 2; }
[ -n "$TARGET" ]    || { echo "--target required"; exit 2; }
LD_REF="${LD_REF:-$TARGET}"

mkdir -p "$OUTDIR"

die() { echo; echo "FATAL: $1" >&2; exit 1; }

# data rows (excluding header), never negative
rows() {
  if [ -s "$1" ]; then
    n=$(wc -l < "$1" | tr -d ' ')
    [ "$n" -gt 0 ] && echo $(( n - 1 )) || echo 0
  else
    echo 0
  fi
}
lines() { [ -s "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
stage() { printf "  %-46s %10s\n" "$1" "$2"; }

# Guard: fail at the stage that emptied the file, not three stages later.
require() {  # file, stage label
  n=$(rows "$1")
  stage "$2" "$n"
  [ "$n" -gt 0 ] || die "stage '$2' left 0 variants. Everything downstream would be
empty. Inspect $1 and the logs in $OUTDIR."
}

# Drop rows whose column 3 appears in an ID list. Empty list = keep everything.
exclude_by_id() {  # id_list, input, output
  if [ -s "$1" ]; then
    awk 'BEGIN{FS=OFS="\t"}
         NR==FNR && FILENAME==ARGV[1] {bad[$1]=1; next}
         FNR==1 || !($3 in bad)' "$1" "$2" > "$3"
  else
    cp "$2" "$3"
  fi
}

# Keep only rows whose column 3 appears in an ID list.
keep_by_id() {  # id_list, input, output
  [ -s "$1" ] || die "keep list $1 is empty -- nothing would survive"
  awk 'BEGIN{FS=OFS="\t"}
       NR==FNR && FILENAME==ARGV[1] {ok[$1]=1; next}
       FNR==1 || ($3 in ok)' "$1" "$2" > "$3"
}

# Fail on a missing fileset here, with the path, rather than as an opaque awk or
# plink error three stages in.
for pfx_label in "target:$TARGET" "LD panel:$LD_REF"; do
  label="${pfx_label%%:*}"; pfx="${pfx_label#*:}"
  for ext in bed bim fam; do
    [ -f "$pfx.$ext" ] || {
      echo "FATAL: $label fileset incomplete -- $pfx.$ext not found" >&2
      echo "       (CWD is $(pwd))" >&2
      exit 2; }
  done
done
[ -f "$BASE_GWAS" ] || { echo "FATAL: base GWAS not found: $BASE_GWAS (CWD is $(pwd))" >&2; exit 2; }


echo
echo "PRS-Net base QC"
echo "=============================================================="
echo "  base GWAS : $BASE_GWAS"
echo "  target    : $TARGET"
echo "  LD ref    : $LD_REF"
[ "$LD_REF" = "$TARGET" ] && echo "              (same as target -- intersection is a no-op, as intended)"
echo "  outdir    : $OUTDIR"
echo
echo "Variant accounting"
echo "--------------------------------------------------------------"

# ---- 0. append BETA = log(OR) -----------------------------------------
PREPARED="$OUTDIR/base_gwas.prepared.tsv"
if [ -n "$SKIP_PREPARE" ]; then
  cp "$BASE_GWAS" "$PREPARED"
  require "$PREPARED" "input (prepare skipped)"
else
  NCOL=$(head -1 "$BASE_GWAS" | awk -F'\t' '{print NF}')
  if [ "$NCOL" -eq 12 ]; then
    cp "$BASE_GWAS" "$PREPARED"
    require "$PREPARED" "input (BETA already present)"
  else
    python "$SCRIPT_DIR/prepare_base_gwas.py" \
        --input "$BASE_GWAS" --output "$PREPARED" \
        --effect_allele "$EFFECT_ALLELE" $STRIP_CHR --drop_invalid \
        > "$OUTDIR/prepare_base.log" 2>&1 \
        || { tail -20 "$OUTDIR/prepare_base.log"; die "prepare_base_gwas.py failed"; }
    require "$PREPARED" "after BETA=log(OR) + validity filter"
  fi
fi

# ---- 1. MAF / INFO ----------------------------------------------------
awk -v mm="$MIN_MAF" -v mi="$MIN_INFO" 'BEGIN{OFS="\t"}
  NR==1 {print; next}
  ($11+0 > mm && $10+0 > mi) {print}' "$PREPARED" > "$OUTDIR/gwas.a1.txt"
require "$OUTDIR/gwas.a1.txt" "after MAF>$MIN_MAF and INFO>$MIN_INFO"

# ---- 2. harmonise SNP ids to the target .bim --------------------------
awk 'BEGIN{FS=OFS="\t"}
  NR==FNR && FILENAME==ARGV[1] {a[$1,$4]=$2; next}
  FNR==1  {print; next}
  ($1,$2) in a {$3=a[$1,$2]; print}' "$TARGET.bim" "$OUTDIR/gwas.a1.txt" > "$OUTDIR/gwas.a2.txt"
require "$OUTDIR/gwas.a2.txt" "after (CHR,BP) match to target .bim"

# ---- 3. duplicates ----------------------------------------------------
awk '{seen[$3]++; if (seen[$3]==1) print}' "$OUTDIR/gwas.a2.txt" > "$OUTDIR/gwas.a3.txt"
require "$OUTDIR/gwas.a3.txt" "after duplicate-ID removal"

# ---- 4. strand-ambiguous ---------------------------------------------
awk '!( ($4=="A" && $5=="T") || ($4=="T" && $5=="A") ||
        ($4=="G" && $5=="C") || ($4=="C" && $5=="G") )' \
    "$OUTDIR/gwas.a3.txt" > "$OUTDIR/gwas.a4.txt"
require "$OUTDIR/gwas.a4.txt" "after strand-ambiguous removal"

# ---- 5. target / LD variant QC + allele mismatch -----------------------
# mismatch.R reads <prefix>.snplist, which upstream never creates.
$PLINK --bfile "$TARGET" --maf "$T_MAF" --hwe "$T_HWE" --geno "$T_GENO" --mind "$T_MIND" \
       --write-snplist --make-just-fam --out "$OUTDIR/target" --memory "$MEM" \
       > "$OUTDIR/plink_target_qc.log" 2>&1
[ -s "$OUTDIR/target.snplist" ] || { tail -20 "$OUTDIR/plink_target_qc.log"; die "target QC produced no snplist"; }
stage "target variants surviving QC" "$(lines "$OUTDIR/target.snplist")"

$PLINK --bfile "$LD_REF" --maf "$T_MAF" --hwe "$T_HWE" --geno "$T_GENO" --mind "$T_MIND" \
       --write-snplist --make-just-fam --out "$OUTDIR/ldref" --memory "$MEM" \
       > "$OUTDIR/plink_ld_qc.log" 2>&1
[ -s "$OUTDIR/ldref.snplist" ] || die "LD panel QC produced no snplist"

# mismatch.R expects <bfile>.bim beside <bfile>.snplist
TARGET_ABS=$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")
LDREF_ABS=$(cd "$(dirname "$LD_REF")" && pwd)/$(basename "$LD_REF")
# rm first: an earlier version symlinked these, and cp onto a symlink that
# resolves to the source is a no-op that BSD cp reports as "identical".
rm -f "$OUTDIR/target.bim" "$OUTDIR/ldref.bim"
cp "$TARGET_ABS.bim" "$OUTDIR/target.bim"
cp "$LDREF_ABS.bim"  "$OUTDIR/ldref.bim"

Rscript "$SCRIPT_DIR/mismatch.R" "$OUTDIR/target" "$OUTDIR/gwas.a4.txt" 'target_data' "$OUTDIR/" \
        > "$OUTDIR/mismatch_target.log" 2>&1 \
        || { tail -20 "$OUTDIR/mismatch_target.log"; die "mismatch.R (target) failed"; }
Rscript "$SCRIPT_DIR/mismatch.R" "$OUTDIR/ldref"  "$OUTDIR/gwas.a4.txt" 'ld_data'     "$OUTDIR/" \
        > "$OUTDIR/mismatch_ld.log" 2>&1 \
        || { tail -20 "$OUTDIR/mismatch_ld.log"; die "mismatch.R (LD) failed"; }
[ -f "$OUTDIR/target_data.a1" ] || die "mismatch.R did not write target_data.a1"

exclude_by_id "$OUTDIR/target_data.tofilter.snplist" "$OUTDIR/gwas.a4.txt" "$OUTDIR/gwas.a5.txt"
exclude_by_id "$OUTDIR/ld_data.tofilter.snplist"     "$OUTDIR/gwas.a5.txt" "$OUTDIR/gwas.a6.txt"
require "$OUTDIR/gwas.a6.txt" "after allele-mismatch resolution"

# ---- 6. differential missingness --------------------------------------
: > "$OUTDIR/missdiff_filter.txt"
if [ -n "$PHENO" ] && [ -f "$PHENO" ]; then
  # --allow-no-sex is load-bearing: plink sets the phenotype to missing for every
  # ambiguous-sex sample by default. A .fam with no sex column therefore yields
  # zero cases and zero controls, and --test-missing silently skips itself even
  # though the phenotype is perfectly valid.
  $PLINK --bfile "$TARGET" --pheno "$PHENO" --allow-no-sex \
         --test-missing midp --pfilter 1e-5 \
         --out "$OUTDIR/MISSDIFF" --memory "$MEM" > "$OUTDIR/plink_missing.log" 2>&1
  if [ -f "$OUTDIR/MISSDIFF.missing" ]; then
    tail -n +2 "$OUTDIR/MISSDIFF.missing" | awk 'NF>1 {print $2}' > "$OUTDIR/missdiff_filter.txt"
    echo "  [INFO] differential-missingness flagged $(lines "$OUTDIR/missdiff_filter.txt") variants"
  else
    echo "  [WARN] --test-missing produced no output; skipping this filter."
    echo "         plink said:"
    grep -iE "error|warning" "$OUTDIR/plink_missing.log" | head -3 | sed 's/^/           /'
    echo "         --test-missing needs a binary case/control phenotype coded 1/2"
    echo "         (or 0/1 with --1), and both classes must survive plink's"
    echo "         sex handling. Check $PHENO and $OUTDIR/plink_missing.log."
  fi
else
  echo "  [WARN] no --pheno given; skipping the differential-missingness filter"
fi
exclude_by_id "$OUTDIR/missdiff_filter.txt" "$OUTDIR/gwas.a6.txt" "$OUTDIR/gwas.a7.txt"
require "$OUTDIR/gwas.a7.txt" "after differential-missingness filter"

# ---- 7. target n LD intersection --------------------------------------
awk 'NR>1 {print $3}' "$OUTDIR/gwas.a7.txt" > "$OUTDIR/gwas.snplist"
[ -s "$OUTDIR/gwas.snplist" ] || die "gwas.snplist is empty"

$PLINK --bfile "$LD_REF" --extract "$OUTDIR/gwas.snplist" --a1-allele "$OUTDIR/ld_data.a1" \
       --make-just-bim --out "$OUTDIR/ld_data.SNP" --memory "$MEM" >> "$OUTDIR/plink_ld_qc.log" 2>&1
[ -f "$OUTDIR/ld_data.SNP.bim" ] || { tail -20 "$OUTDIR/plink_ld_qc.log"; die "LD --make-just-bim failed"; }

$PLINK --bfile "$TARGET" --extract "$OUTDIR/gwas.snplist" --a1-allele "$OUTDIR/target_data.a1" \
       --make-just-bim --out "$OUTDIR/target_data.SNP" --memory "$MEM" >> "$OUTDIR/plink_target_qc.log" 2>&1
[ -f "$OUTDIR/target_data.SNP.bim" ] || { tail -20 "$OUTDIR/plink_target_qc.log"; die "target --make-just-bim failed"; }

awk 'NR==FNR && FILENAME==ARGV[1] {s[$2]=1; next} $2 in s {print $2}' \
    "$OUTDIR/ld_data.SNP.bim" "$OUTDIR/target_data.SNP.bim" > "$OUTDIR/common_snps.txt"
sort "$OUTDIR/common_snps.txt" > "$OUTDIR/_c.s"
sort "$OUTDIR/gwas.snplist"    > "$OUTDIR/_g.s"
comm -12 "$OUTDIR/_c.s" "$OUTDIR/_g.s" > "$OUTDIR/snplist"
rm -f "$OUTDIR/_c.s" "$OUTDIR/_g.s"
stage "final intersected SNP list" "$(lines "$OUTDIR/snplist")"
[ -s "$OUTDIR/snplist" ] || die "intersection is empty -- target and LD .bim share no variant IDs"

$PLINK --bfile "$TARGET" --extract "$OUTDIR/snplist" --a1-allele "$OUTDIR/target_data.a1" \
       --make-bed --out "$OUTDIR/target_data.PH" --memory "$MEM" >> "$OUTDIR/plink_target_qc.log" 2>&1
[ -f "$OUTDIR/target_data.PH.bed" ] || { tail -20 "$OUTDIR/plink_target_qc.log"; die "target_data.PH not written"; }

keep_by_id "$OUTDIR/snplist" "$OUTDIR/gwas.a7.txt" "$OUTDIR/gwas.QC.txt"
awk 'NR==1 {print "SNP","P"; next} {print $3, $8}' "$OUTDIR/gwas.QC.txt" > "$OUTDIR/SNP.pvalue"
require "$OUTDIR/gwas.QC.txt" "gwas.QC.txt"

echo "--------------------------------------------------------------"
echo
echo "Outputs in $OUTDIR:"
echo "  target_data.PH.{bed,bim,fam}   $(lines "$OUTDIR/target_data.PH.fam") samples, $(lines "$OUTDIR/target_data.PH.bim") variants"
echo "  gwas.QC.txt                    $(rows "$OUTDIR/gwas.QC.txt") variants"
echo "  SNP.pvalue"
echo
echo "Next: bash bin/prsnet/test/test_pipeline_smoke.sh --base_dir . \\"
echo "        --target $OUTDIR/target_data.PH --chrom 22"