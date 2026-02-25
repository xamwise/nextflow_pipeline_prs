process uni_lasso {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/uni_lasso", mode: 'copy'

    input:
    val data_prefix
    val pheno_file
    val cov_file
    val pcs_file
    val n_folds
    val out_dir

    output:
    path('uni_lasso*.html'), emit: uni_lasso, optional: true

    script:
 
    
    """
    mkdir -p ${out_dir}

    python ${params.base_dir}/bin/uni_lasso.py \\
        --data_prefix $data_prefix \\
        --pheno_file $pheno_file \\
        --cov_file $cov_file \\
        --pcs_file $pcs_file \\
        --n_folds $n_folds \\
        --out_dir $out_dir \\
   
    """
