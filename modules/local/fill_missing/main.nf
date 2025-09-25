process fill_missing{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/fill_missing", mode: 'copy'

    input:
    val input
    val out
 
    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --fill-missing-a2 \\
    --make-bed \\
    --out $out \\
    """
}
