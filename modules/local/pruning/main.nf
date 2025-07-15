process pruning{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/pruning", mode: 'copy'

    input:
    val placeholder
    val input
    val keep
    val extract
    val out
  
    output:
    val out

    script:
    """
    plink  --bfile $input \\
            --keep $keep \\
            --extract $extract \\
            --indep-pairwise 200 50 0.25 \\
            --out $out
    """
}