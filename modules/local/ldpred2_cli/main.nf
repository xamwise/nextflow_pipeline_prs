process ldpred2_cli {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/ldpred2_cli", mode: 'copy'
    container "${params.base_dir}/data/supplement_data/LDpred2/containers/containers/latest/r.sif"

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
    val data
    val qc_prefix

    output:
    val out

    script:
    // def scripts_dir  = "${params.base_dir}/data/supplement_data/LDpred2/containers/scripts/pgs/LDpred2"
    def scripts_dir  = "${params.base_dir}/bin/ldpred2"
    def ldref_dir    = "${params.base_dir}/data/supplement_data/LDpred2/ldpred2_ref"
    def geno_in      = "${bed}.bed"
    def geno_rds     = "${qc_prefix}/${data}/tmp-data/ldpred2_genotype.rds"
    def stat_type    = (trait == "binary" || trait == "bin") ? "OR" : "BETA"
    def ldpred_mode  = (model == "auto") ? "auto" : "inf"
    def out_suffix   = (model == "auto") ? "auto" : "inf"
    """
    mkdir -p ${qc_prefix}/${data}/tmp-data
    mkdir -p ${out}

    # Convert genotype to bigSNP format
    Rscript ${scripts_dir}/createBackingFile.R \\
        --file-input ${geno_in} \\
        --file-output ${geno_rds}

    # Run LDpred2
    Rscript ${scripts_dir}/ldpred2.R \\
        --ldpred-mode ${ldpred_mode} \\
        --merge-by-rsid \\
        --col-stat ${stat_type} \\
        --col-stat-se SE \\
        --stat-type ${stat_type} \\
        --geno-file-rds ${geno_rds} \\
        --sumstats ${sum_stats} \\
        --ld-file ${ldref_dir}/ldref_hm3_plus/LD_with_blocks_chr@.rds \\
        --ld-meta-file ${ldref_dir}/map_hm3_plus.rds \\
        --out ${out}/ldpred2_cli.${out_suffix}
    """
}