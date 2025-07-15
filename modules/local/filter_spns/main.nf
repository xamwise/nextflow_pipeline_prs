process filter_spns{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/filter_spns", mode: 'copy'

    input:
    val input
    val out
    val extract

    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --extract $extract \\
    --make-bed \\
    --out $out 
    """
}