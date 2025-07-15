process split_data{å

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/filter_spns", mode: 'copy'

    input:
    val input
    val out
    val test
    val validation

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