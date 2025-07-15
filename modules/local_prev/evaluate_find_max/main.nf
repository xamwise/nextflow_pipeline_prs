process EVALUATE_FIND_MAX {
    tag "${test_mode}_${model_name}_${split_id}"
    label 'process_single'

    input:
    tuple val(model_name), val(test_mode), val(split_id), path(hpam_yamls), path(pred_datas)
    val metric

    output:
    tuple val(model_name), val(split_id), val(test_mode), path('best_hpam_combi_*.yaml'), emit: best_combis

    script:
    """
    evaluate_and_find_max.py \\
        --model_name "${model_name}" \\
        --split_id $split_id \\
        --hpam_yamls $hpam_yamls \\
        --pred_datas $pred_datas \\
        --metric $metric
    """

}
