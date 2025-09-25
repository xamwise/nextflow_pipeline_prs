process prset {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/prset", mode: 'copy'
    
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
    val gtf
    val set

    output:
    val out

    script:
    """
    mkdir -p ${out}

    Rscript ${params.base_dir}/bin/PRSice.R \\
        --prsice ${params.base_dir}/bin/PRSice_mac \\
        --base $base  \
        --target $target \\
        --A1 $a1 \\
        --A2 $a2 \\
        --stat $stat \\
        --pheno $pheno \\
        --cov $cov \\
        --binary-target $binary_target \\
        --base-maf $base_maf \\
        --base-info $base_info \\
        --out $out \\
        --gtf $gtf \\
        --msigdb $set
    """

}