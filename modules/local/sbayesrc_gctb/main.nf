/*
 * SBayesRC with the GCTB command line tool, following the GCTB SBayesRC tutorial
 * (https://gctbhub.cloud.edu.au/software/gctb/#SBayesRCTutorial). Second implementation next to
 * modules/local/sbayesr, which runs SBayesRC with the SBayesRC R package.
 *   1. QC and imputation of the summary statistics with the eigen-decomposed LD reference
 *   2. SBayesRC with functional annotations: SNP effects in sbayesrc.snpRes (Name, A1, A1Effect)
 *   3. PRS of the target samples with PLINK: sbayesrc.profile
 */
process sbayesrc_gctb {

    label 'process_single'

    input:
    val ma_file
    val ld_folder
    val annot
    val bed
    val out_dir

    output:
    val out_dir

    script:
    """
    mkdir -p ${out_dir}

    ${params.base_dir}/bin/gctb \\
        --ldm-eigen ${ld_folder} \\
        --gwas-summary ${ma_file} \\
        --impute-summary \\
        --thread ${task.cpus} \\
        --out ${out_dir}/sbayesrc

    ${params.base_dir}/bin/gctb \\
        --ldm-eigen ${ld_folder} \\
        --gwas-summary ${out_dir}/sbayesrc.imputed.ma \\
        --sbayes RC \\
        --annot ${annot} \\
        --thread ${task.cpus} \\
        --out ${out_dir}/sbayesrc

    plink --bfile ${bed} --score ${out_dir}/sbayesrc.snpRes 2 5 8 header sum center --out ${out_dir}/sbayesrc
    """
}
