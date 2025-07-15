process combine_cov{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/combine_cov", mode: 'copy'

    input:
    val cov
    val pcs
    val out

    output:
    val out

    script:
    """
    Rscript /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/combine_cov.R --cov $cov --pcs $pcs --out $out
    """
}