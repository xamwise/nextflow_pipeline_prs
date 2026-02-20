process filter_by_fold{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/filter_by_fold", mode: 'copy'

    input:
    val input
    val  n_folds
    val output_dir
    val random_state
  
    output:
    val output_dir

    script:
    """
    python ${params.base_dir}/bin/create_folds.py --pheno_file $input --n_folds $n_folds --output_dir $output_dir --random_state $random_state
    """
}