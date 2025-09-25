process lassosum2{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/lassosum2", mode: 'copy'

    input:
    val bed
    val pheno
    val cov
    val pcs
    val sum_stats
    val trait
    val sample_size
    val out

    output:
    val out
    
    script:
    """
    mkdir -p ${out}
    Rscript ${params.base_dir}/bin/lassosum2.R \
        --bed $bed \
        --sum_stats $sum_stats \
        --pheno $pheno \
        --cov $cov \
        --pcs $pcs \
        --trait $trait \
        --n_samples $sample_size \
        --relax_qc 'TRUE' \
        --out $out/lassosum2
    """
}


// Binary phenotype example

//    """
//     Rscript ${params.base_dir}/bin/lassosum2.R \
//         --bed $bed \
//         --sum_stats $sum_stats \
//         --pheno $pheno \
//         --cov $cov \
//         --pcs $pcs \
//         --trait binary \
//         --n_cases 20791 \
//         --n_controls 323124 \
//         --out $out
//     """
