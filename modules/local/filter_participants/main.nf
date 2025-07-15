process filter_participants{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/filter_participants", mode: 'copy'

    input:
    val input
    val out
    val keep

    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --keep $keep \\
    --make-bed \\
    --out $out 
    """
}