process prepare_fold_data {
    tag "fold_${fold_id}"
    label 'process_medium'
    publishDir "${output_base_dir}", mode: 'copy'

    input:
    tuple val(fold_id), val(test_ids_file)
    val qc_prefix
    val pheno_file
    val cov_file
    val pcs_file
    val output_base_dir

    output:
    tuple val(fold_id),
          val("${output_base_dir}/train_folds/fold_${fold_id}/data"),
          val("${output_base_dir}/test_folds/fold_${fold_id}/data"),
          val("${output_base_dir}/train_folds/fold_${fold_id}/pheno.tsv"),
          val("${output_base_dir}/test_folds/fold_${fold_id}/pheno.tsv"),
          val("${output_base_dir}/train_folds/fold_${fold_id}/cov.tsv"),
          val("${output_base_dir}/test_folds/fold_${fold_id}/cov.tsv"),
          val("${output_base_dir}/train_folds/fold_${fold_id}/pcs.tsv"),
          val("${output_base_dir}/test_folds/fold_${fold_id}/pcs.tsv"),
          emit: fold_data

    script:
    """
    set -euo pipefail

    TRAIN_DIR=${output_base_dir}/train_folds/fold_${fold_id}
    TEST_DIR=${output_base_dir}/test_folds/fold_${fold_id}
    mkdir -p \$TRAIN_DIR \$TEST_DIR

    # Derive train IDs = everyone in .fam not in the test-ID file (match on IID)
    awk 'NR==FNR{t[\$2]=1; next} !(\$2 in t){print \$1, \$2}' \\
        ${test_ids_file} ${qc_prefix}.fam > train_ids.txt

    # Filter genotypes
    plink --bfile ${qc_prefix} --keep train_ids.txt      --make-bed --out \$TRAIN_DIR/data
    plink --bfile ${qc_prefix} --keep ${test_ids_file}   --make-bed --out \$TEST_DIR/data

    # Filter pheno / cov / PCs (FID IID assumed as first two cols)
    python ${params.base_dir}/bin/filter_by_ids.py \\
        --input ${pheno_file} --train_ids train_ids.txt --test_ids ${test_ids_file} \\
        --train_out \$TRAIN_DIR/pheno.tsv --test_out \$TEST_DIR/pheno.tsv

    python ${params.base_dir}/bin/filter_by_ids.py \\
        --input ${cov_file}   --train_ids train_ids.txt --test_ids ${test_ids_file} \\
        --train_out \$TRAIN_DIR/cov.tsv   --test_out \$TEST_DIR/cov.tsv

    python ${params.base_dir}/bin/filter_by_ids.py \\
        --input ${pcs_file}   --train_ids train_ids.txt --test_ids ${test_ids_file} \\
        --train_out \$TRAIN_DIR/pcs.tsv   --test_out \$TEST_DIR/pcs.tsv
    """
}