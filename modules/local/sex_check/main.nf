process sex_check{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sex_check", mode: 'copy'

    input:
    val placeholder
    val input
    val out
    val extract
    val keep
    val update_in
    val update_out

    output:
    val out

    script:
    """
    plink \\
    --bfile $input \\
    --extract $extract \\
    --keep $keep \\
    --check-sex \\
    --out $out

    Rscript /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/valid_sex_update.R --file $keep --sex_check $update_in --out $update_out
    """
}