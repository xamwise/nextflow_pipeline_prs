/*
Input files

Plink files. One plink file (binary PED format) per chromosome from the validation (insert the character "[1:22]" instead of the chromsome numbers). These are used as LD-reference panel and later to compute PRS.
Example: plinkfile="my_plink_files/Final.chrom[1:22]"
Use flag: --gf
File with functional enrichments (see Gazal et al 2017 Nat Genet and Finucane et al 2015 Nat Genet).
First you will need to estimate the per-SNP heritability inferred using S-LDSC (see instructions here) under the baselineLD model (you can download Baseline-LD annotations here), and use the summary statistics that will be later used as training. When running S-LDSC make sure to use the --print-coefficients flag to get the regression coefficients.
After running S-LDSC:
Get h2g estimate from the *.log file.
Get the regression coefficients from the *.results file (column 8). Divide the regression coeffients by h2g, define it as T=tau/h2g which is a vector of dimension Cx1, where C is the total number of annotations.
From the baselineLD annotations downloaded from here, read the annotations file baselineLD.*.annot.gz, and only keep the annotations columns (i.e. remove first 4 columns). Call this matrix X, with dimensions MxC, where M is the number of SNPs and C is the total number of annotations.
Define the expected per-SNP heritability as a Mx1 vector (sigma2_i from the LDpred-funct manuscript) as the result from multiplying the matrix X times T.
Format of FUNCTFILE:

Column 1: SNP ID
Column 2: per-SNP heritability
Use flag: --FUNCT_FILE

Summary statistics file. Please check that the summary statistics contains a column for each of the following field (header is important here, important fields are highlighted in bold font, the order of the columns it is not important).
CHR Chromosome
SNP SNP ID
BP Physical position (base-pair)
A1 Minor allele name (based on whole sample)
A2 Major allele name
P Asymptotic p-value
BETA Effect size
Z Z-score (default). If instead of Z-score the Chi-square statistic is provided, use the flag --chisq, and CHISQ as column field.
Use flag: --ssf
Phenotype file.
Format:

Column 1: FID
Column 2: phenotype
This file doesn't have a header.

Use flag: --pf

Input parameters:

Training sample size.
Use flag: --N
Estimated SNP Heritability (pre-compute this using your favorite method).
Use flag: --H2
LD radius (optional). If not provided, it is computed as (1/2)*0.15% of total number of SNPs.
Use flag: --ld_radius
Output files

Coordinated files: This is an hdf5 file that stores the coordinated genotype data with the summary statistics and functional enrichments.
Use flag: --coord
Note: the output file needs to be named differently for different runs.
Posterior mean effect sizes: Estimated posterior mean effect size from LDpred-funct-inf.
Use flag: --posterior_means
Output: Polygenic risk score for each individual in the validation.
Description:

Column 1: Sample ID
Column 2: True phenotype
Column 3: PRS using all-snps and marginal effect sizes.
Colunm 4: PRS obtained using LD-pred-funct-inf
Column 5-K: PRS(k) defined in equation 5 from Marquez-Luna, et al, Biorxiv.
Use flag: --out

*/


process ldpred_funct {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/ldpred_funct", mode: 'copy'

    input:
    val plinkfile //"my_plink_files/Final.chrom[1:22]"
    val outCoord //"my_analysis/Coord_Final"
    val statsfile //"my_analysis/sumary_statistics.txt"
    val functfile //"my_analysis/functfile_sldsc.txt"
    val outLdpredfunct //"my_analysis/ldpredfunct_posterior_means"
    val outValidate //"my_analysis/ldpredfunct_prs"
    val phenotype //"my_analysis/trait_1.txt"
    val N //training sample size
    val h2 // pre-computed SNP heritability
    
    //optional 
    val ld_radius // Pre-defined ld-radius
    val K // Number of bins for LDpred-funct


    output:
    path('ldpred_funct*.html'), emit: ldpred_funct, optional: true

    script:
    """
    python /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/ldpredfunct.py \\
        --gf $plinkfile \\
        --pf $phenotype \\
        --FUNCT_FILE $functfile \\
        --coord $outCoord \\
        --ssf $statsfile \\
        --N $N \\
        --posterior_means $outLdpredfunct \\
        --H2 $h2 \\
        --out $outValidate > ${outValidate}.log
    """

}