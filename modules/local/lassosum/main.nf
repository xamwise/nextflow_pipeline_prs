process lassosum{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/lassosum", mode: 'copy'

    input:
    val bed
    val pheno
    val cov
    val pcs
    val sum_stats
    val out

    output:
    val out

    script:
    """
    Rscript /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/lassosum.R --bed $bed --pheno $pheno --cov $cov --pcs $pcs --sum_stats $sum_stats --out $out
    """

}