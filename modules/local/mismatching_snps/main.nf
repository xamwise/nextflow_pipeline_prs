process mismatching_snps{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/mismatching_snps", mode: 'copy'

    input:
    val placeholder
    val input
    val sum_stats
    val snp_list
    val mismatch
    val out
  
    output:
    val out

    script:
    """
    Rscript /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/mismatching_snps.R --file $input --sumstats $sum_stats --snplist $snp_list --mismatch $mismatch --out $out
    """
}