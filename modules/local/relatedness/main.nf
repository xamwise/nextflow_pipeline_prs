process relatedness{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/relatedness", mode: 'copy'

    input:
    val placeholder
    val input
    val out
    val extract
    val keep
    val cutoff

    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --extract $extract \\
    --keep $keep \\
    --rel-cutoff $cutoff \\
    --out $out
    """
}