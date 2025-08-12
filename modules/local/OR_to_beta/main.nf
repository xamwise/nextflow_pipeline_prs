process OR_to_beta{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/OR_to_beta", mode: 'copy'

    input:
    val input
    val out

    output:
    val out

    script:
    """
    Rscript ${params.base_dir}/bin/OR_to_beta.R --input $input --out $out
    """
}