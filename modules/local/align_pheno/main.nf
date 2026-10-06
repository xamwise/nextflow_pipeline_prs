process align_pheno {

    label 'process_single'

    input:
    val pheno
    val fam
    val out

    output:
    val out

    script:
    """
    python ${params.base_dir}/bin/align_pheno.py --pheno_file $pheno --fam_file $fam --out $out
    """
}
