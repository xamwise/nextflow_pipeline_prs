process prs_cs_preprocess {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/prs_cs_preprocess", mode: 'copy'

    input:
    val input
    val out

    output:
    val out

    script:
    """
    python /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/prs_cs_preprocess.py \\
    --input $input \\
    --out $out 
    """
}