process sbayes_cojo {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sbayes_cojo", mode: 'copy'

    input:
    val input
    val out

    output:
    val out

    script:
    """
    python ${params.base_dir}/bin/sbayes_cojo.py \\
    --input $input \\
    --out $out 
    """
}