process beta_to_OR{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/beta_to_OR", mode: 'copy'

    input:
    val input
    val out

    output:
    val out

    script:
    """
    Rscript /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/beta_to_OR.R --input $input --out $out
    """
}