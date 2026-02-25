process remove_participants{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/remove_participants", mode: 'copy'

    input:
    val input
    val out
    val remove

    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --remove $remove \\
    --make-bed \\
    --out $out 
    """
}