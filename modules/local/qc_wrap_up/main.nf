process qc_wrap_up{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/qc_wrap_up", mode: 'copy'

    input:
    val placeholder
    val input
    val out
    val extract
    val keep
    val exclude
    val a1

    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --make-bed \\
    --keep $keep \\
    --out $out \\
    --extract $extract \\
    --exclude $exclude \\
    --a1-allele $a1
    """
}