process gctb_ma {

    label 'process_single'

    input:
    val sum_stats
    val snp_info
    val out

    output:
    val out

    script:
    """
    python ${params.base_dir}/bin/sumstats_to_gctb_ma.py --input $sum_stats --snp_info $snp_info --out $out
    """
}
