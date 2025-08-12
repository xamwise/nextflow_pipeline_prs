process combine_cov{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/combine_cov", mode: 'copy'

    input:
    val cov
    val pcs
    val out

    output:
    val out

    script:
    """
    Rscript ${params.base_dir}/bin/combine_cov.R --cov $cov --pcs $pcs --out $out
    """
}