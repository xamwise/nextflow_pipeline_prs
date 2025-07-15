/*
Using PRS-CSx

python PRScsx.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --pop=POPULATION --out_dir=OUTPUT_DIR --out_name=OUTPUT_FILE_PREFIX [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR --chrom=CHROM --meta=META_FLAG --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED]

PATH_TO_REFERENCE (required): Full path to the directory that contains the SNP information file and LD reference panels. If the 1000 Genomes reference is used, the folder would contain the SNP information file snpinfo_mult_1kg_hm3 and one or more of the LD reference files: ldblk_1kg_afr, ldblk_1kg_amr, ldblk_1kg_eas, ldblk_1kg_eur, ldblk_1kg_sas; if the UK Biobank reference is used, the folder would contain the SNP information file snpinfo_mult_ukbb_hm3 and one or more of the LD reference files: ldblk_ukbb_afr, ldblk_ukbb_amr, ldblk_ukbb_eas, ldblk_ukbb_eur, ldblk_ukbb_sas.

VALIDATION_BIM_PREFIX (required): Full path and the prefix of the bim file for the target (validation/testing) dataset. This file is used to provide a list of SNPs that are available in the target dataset.

SUM_STATS_FILE (required): Full path and the file name of the GWAS summary statistics. Multiple GWAS summary statistics files are allowed and should be separated by comma. The summary statistics file must include either BETA/OR + SE or BETA/OR + P. When using BETA/OR + SE as the input, the file must have the following format (including the header line):

    SNP          A1   A2   BETA      SE
    rs4970383    C    A    -0.0064   0.0090
    rs4475691    C    T    -0.0145   0.0094
    rs13302982   A    G    -0.0232   0.0199
    ...
Or:

    SNP          A1   A2   OR        SE
    rs4970383    A    C    0.9825    0.0314                 
    rs4475691    T    C    0.9436    0.0319
    rs13302982   A    G    1.1337    0.0543
    ...
where SNP is the rs ID, A1 is the effect allele, A2 is the alternative allele, BETA/OR is the effect/odds ratio of the A1 allele, SE is the standard error of the effect. Note that when OR is used, SE corresponds to the standard error of logOR.

When using BETA/OR + P as the input, the file must have the following format (including the header line):

    SNP          A1   A2   BETA      P
    rs4970383    C    A    -0.0064   0.4778
    rs4475691    C    T    -0.0145   0.1245
    rs13302982   A    G    -0.0232   0.2429
    ...
Or:

    SNP          A1   A2   OR        P
    rs4970383    A    C    0.9825    0.5737                 
    rs4475691    T    C    0.9436    0.0691
    rs13302982   A    G    1.1337    0.0209
    ...
where SNP is the rs ID, A1 is the effect allele, A2 is the alternative allele, BETA/OR is the effect/odds ratio of the A1 allele, P is the p-value of the effect. Here, a standardized effect size is calculated using the p-value while BETA/OR is only used to determine the direction of an association. Therefore if z-scores or even +1/-1 indicating effect directions are presented in the BETA column, the algorithm should still work properly.

GWAS_SAMPLE_SIZE (required): Sample sizes of the GWAS, in the same order of the GWAS summary statistics files, separated by comma.

POPULATION (required): Population of the GWAS sample, in the same order of the GWAS summary statistics files, separated by comma. For both the 1000 Genomes reference and the UK Biobank reference, AFR, AMR, EAS, EUR and SAS are allowed.

OUTPUT_DIR (required): Output directory of the posterior effect size estimates.

OUTPUT_FILE_PREFIX (required): Output filename prefix of the posterior effect size estimates.

PARAM_A (optional): Parameter a in the gamma-gamma prior. Default is 1.

PARAM_B (optional): Parameter b in the gamma-gamma prior. Default is 0.5.

PARAM_PHI (optional): Global shrinkage parameter phi. If phi is not specified, it will be learnt from the data using a fully Bayesian approach. This usually works well for polygenic traits with very large GWAS sample sizes (hundreds of thousands of subjects). For GWAS with limited sample sizes (including most of the current disease GWAS), fixing phi to 1e-2 (for highly polygenic traits) or 1e-4 (for less polygenic traits), or doing a small-scale grid search (e.g., phi=1e-6, 1e-4, 1e-2, 1) to find the optimal phi value in the validation dataset often improves perdictive performance.

MCMC_ITERATIONS (optional): Total number of MCMC iterations. Default is 1,000 * the number of discovery populations.

MCMC_BURNIN (optional): Number of burnin iterations. Default is 500 * the number of discovery populations. Both --n_iter and --n_burnin need to be specified to overwrite their default values.

MCMC_THINNING_FACTOR (optional): Thinning factor of the Markov chain. Default is 5.

CHROM (optional): The chromosome on which the model is fitted, separated by comma, e.g., --chrom=1,3,5. Parallel computation for the 22 autosomes is recommended. Default is iterating through 22 autosomes (can be time-consuming).

META_FLAG (optional): If True, return combined SNP effect sizes across populations using an inverse-variance-weighted meta-analysis of the population-specific posterior effect size estimates. Default is False.

WRITE_POSTERIOR_SAMPLES (optional): If True and META_FLAG also True, write all posterior samples of meta-analyzed SNP effect sizes after thinning. Default is False.

SEED (optional): Non-negative integer which seeds the random number generator.

Output

For each input GWAS, PRS-CSx writes posterior SNP effect size estimates for each chromosome to the user-specified directory. The output file contains chromosome, rs ID, base position, A1, A2 and posterior effect size estimate for each SNP. If --meta=True, meta-analyzed posterior effect sizes will also be written to the output directory. An individual-level polygenic score can be produced by concatenating output files from all chromosomes and then using PLINK's --score command (https://www.cog-genomics.org/plink/1.9/score). If polygenic scores are generated by chromosome, use the 'sum' modifier so that they can be combined into a genome-wide score.

In general, there are two approaches to use the output of PRS-CSx. Given a global shrinkage parameter, the first approach calculates one polygenic score for each discovery population using population-specific posterior SNP effect size estimates and learns a linear combination of the polygenic scores that most accurately predicts the trait in the validation dataset. The optimal global shrinkage parameter and linear combination weights are then taken to an independent dataset, where the predictive performance of the final PRS can be assessed. We recommend standardizing the polygenic scores (i.e., converting the scores to zero mean and unit variance) in both validation and testing datasets before learning/applying the linear combination. Alternatively, the 'auto' and 'meta' version of the PRS-CSx algorithm can be used, which does not require a validation dataset to tune hyper-parameters. This approach may be less accurate compared to the linear combination approach (where the final PRS is optimized in a specific population) but can be useful when a validation dataset is not available (e.g., due to limited total sample size in the target dataset).


*/

process prs_csx {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/prs_csx", mode: 'copy'

    input:
    val ref_dir
    val sst_file
    val bim_prefix
    val n_gwas
    val pop
    val out_dir
    val out_name
    // val a
    // val b
    // val phi
    // val n_iter
    // val n_burnin 
    // val thin
    // val chrom
    // val meta
    // val write_pst
    // val seed

    output:
    path('prs_csx*.html'), emit: prs_csx, optional: true

    script:
    """
    python /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/PRScsx.py \\
        --ref_dir $ref_dir \\
        --bim_prefix $bim_prefix \\
        --sst_file $sst_file \\
        --n_gwas $n_gwas \\
        --pop $pop
        --out_dir $out_dir \\
        --out_name $out_name \\
      
    """

    // [--a $a \\
    // --b $b \\
    // --phi $phi \\
    // --n_iter $n_iter \\
    // --n_burnin $n_burnin \\
    // --thin $thin \\
    // --chrom $chrom \\
    // --meta $meta \\
    // --write_pst $write_pst \\
    // --seed $seed]

}