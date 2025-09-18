process sct {
    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sct", mode: 'copy'

    input:
    val ref_dir
    val sst_file
    val pheno_file
    val split
    val out
    val out_dir
    
    output:
    path('sct*.html'), emit: sct, optional: true

    script:
    """
    mkdir -p ${out_dir}

    Rscript ${params.base_dir}/bin/SCT.R \\
    --bed $ref_dir \\
    --sum_stats $sst_file \\
    --pheno $pheno_file \\
    --train_prop $split \\
    --out $out
     """
}