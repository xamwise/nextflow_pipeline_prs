/*
 * Materialises one dataset (fold training set, fold validation set or hold-out test set):
 * PLINK genotypes plus .pheno, .cov, .eigenvec and .covariate restricted to the selected samples.
 *
 * keep_ids / remove_ids are 'FID IID' files with a header (as written by create_folds.py and
 * holdout_split.py), matched on IID. remove_ids is 'NONE' if no samples are to be removed.
 */
process subset_samples {

    tag "fold ${fold_id}: ${split}"
    label 'process_single'

    input:
    tuple val(fold_id), val(split), val(out_dir), val(keep_ids), val(remove_ids)
    val qc_prefix
    val pheno
    val cov
    val pcs
    val covariate
    val population

    output:
    tuple val(fold_id),
          val(split),
          val("${out_dir}/${population}.QC"),
          val("${out_dir}/${population}.pheno"),
          val("${out_dir}/${population}.cov"),
          val("${out_dir}/${population}.eigenvec"),
          val("${out_dir}/${population}.covariate"),
          val(out_dir)

    script:
    def prefix = "${out_dir}/${population}.QC"
    """
    set -euo pipefail
    mkdir -p ${out_dir}

    # bigsnpr backing files and LDpred2 temp data are reused by the R scripts if they exist,
    # so remove those of a previous split to not train on the wrong samples
    rm -f ${prefix}_*.rds ${prefix}_*.bk
    rm -rf ${out_dir}/${population}/tmp-data

    # Turn an ID file into a PLINK keep/remove file, taking FID and IID from the .fam
    ids_to_plink() {
        awk 'NR==FNR{ids[\$2]=1; next} (\$2 in ids){print \$1, \$2}' "\$1" ${qc_prefix}.fam
    }

    ids_to_plink ${keep_ids} > keep.txt
    REMOVE=""
    if [ "${remove_ids}" != "NONE" ]; then
        ids_to_plink ${remove_ids} > remove.txt
        REMOVE="--remove remove.txt"
    fi

    plink \\
    --bfile ${qc_prefix} \\
    --keep keep.txt \\
    \$REMOVE \\
    --make-bed \\
    --out ${prefix}

    # Sample files in the same sample order as the new .fam
    python ${params.base_dir}/bin/subset_by_ids.py --input ${pheno} --fam ${prefix}.fam --out ${out_dir}/${population}.pheno
    python ${params.base_dir}/bin/subset_by_ids.py --input ${cov} --fam ${prefix}.fam --out ${out_dir}/${population}.cov
    python ${params.base_dir}/bin/subset_by_ids.py --input ${pcs} --fam ${prefix}.fam --out ${out_dir}/${population}.eigenvec
    python ${params.base_dir}/bin/subset_by_ids.py --input ${covariate} --fam ${prefix}.fam --out ${out_dir}/${population}.covariate
    """
}
