process heterozygosity{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/heterozygosity", mode: 'copy'

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
    plink   --bfile $input \\
            --extract $extract \\
            --keep $keep \\
            --het \\
            --out $out
    """
}