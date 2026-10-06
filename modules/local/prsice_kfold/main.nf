// Copy of modules/local/prsice that also writes the SNPs of the model (--print-snp -> prsice.snp),
// so the fold models can be applied to the validation and hold-out test samples.
process prsice_kfold {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/prsice", mode: 'copy'

    input:
    val base
    val pheno
    val target
    val cov
    val out
    val a1
    val a2
    val stat
    val binary_target
    val base_maf
    val base_info

    output:
    val out

    script:
    """
    mkdir -p ${out}

    Rscript ${params.base_dir}/bin/PRSice.R \\
        --prsice ${params.base_dir}/bin/PRSice_mac \\
        --base $base  \\
        --target $target \\
        --A1 $a1 \\
        --A2 $a2 \\
        --stat $stat \\
        --pheno $pheno \\
        --cov $cov \\
        --binary-target $binary_target \\
        --base-maf $base_maf \\
        --base-info $base_info \\
        --print-snp \\
        --out ${out}/prsice
    """

}