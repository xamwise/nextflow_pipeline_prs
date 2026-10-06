process holdout_split {

    label 'process_single'

    input:
    val pheno
    val fam
    val test_size
    val output_dir
    val random_state

    output:
    val "${output_dir}/dev.pheno", emit: dev_pheno
    val "${output_dir}/dev_ids.txt", emit: dev_ids
    val "${output_dir}/test_ids.txt", emit: test_ids

    script:
    """
    python ${params.base_dir}/bin/holdout_split.py --pheno_file $pheno --fam_file $fam --test_size $test_size --output_dir $output_dir --random_state $random_state
    """
}
