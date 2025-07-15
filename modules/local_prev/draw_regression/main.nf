process DRAW_REGRESSION {
    tag "${name}_${model}"
    label 'process_single'
    publishDir "${params.outdir}/${params.run_id}/regression_plots", mode: 'copy'

    input:
    tuple val(name), val(model), path(true_vs_pred)

    output:
    path('regression_lines*.html'), emit: regression_lines

    script:
    """
    draw_regression.py \\
    --path_t_vs_p ${true_vs_pred} \\
    --name ${name} \\
    --model ${model}
    """

}
