process ldpred2 {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/ldpred2", mode: 'copy'

    input:
    val bed
    val pheno
    val cov
    val pcs
    val ld
    val sum_stats
    val trait
    val model
    val out

    output:
    val out

    script:
    """
    Rscript ${params.base_dir}/bin/LDpred-2.R --bed $bed --pheno $pheno --cov $cov --pcs $pcs --ld $ld --sum_stats $sum_stats --trait $trait --model $model --out $out 
    """

}