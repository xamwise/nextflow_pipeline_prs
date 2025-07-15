process COLLECT_RESULTS {
    label 'process_single'
    publishDir "${params.outdir}/${params.run_id}", mode: 'copy'

    input:
    path(outfiles)

    output:
    path('evaluation_results.csv'), emit: evaluation_results
    path('evaluation_results_per_drug.csv'), emit: evaluation_results_per_drug
    path('evaluation_results_per_cl.csv'), emit: evaluation_results_per_cl
    path('true_vs_pred.csv'), emit: true_vs_pred

    script:
    """
    collect_results.py \\
        --outfiles $outfiles
    """

}
