#!/usr/bin/env nextflow

/*
 * Quality Control Pipeline for Genetic Data
 * This pipeline performs comprehensive QC on genetic data including:
 * - Basic QC filtering
 * - Summary statistics QC
 * - Pruning and heterozygosity checks
 * - Sample validation
 * - SNP mismatch detection
 * - Sex checks
 * - Relatedness analysis
 * - PCA generation
 */

// Import QC modules
include { quality_control } from '../modules/local/quality_control'
include { quality_control_sum } from '../modules/local/quality_control_sum'
include { pruning } from '../modules/local/pruning'
include { heterozygosity } from '../modules/local/heterozygosity'
include { valid_samples } from '../modules/local/valid_samples'
include { mismatching_snps } from '../modules/local/mismatching_snps'
include { sex_check } from '../modules/local/sex_check'
include { relatedness } from '../modules/local/relatedness'
include { qc_wrap_up } from '../modules/local/qc_wrap_up'
include { pcs } from '../modules/local/pcs'
include { or_to_beta } from '../modules/local/OR_to_beta'
include { beta_to_OR } from '../modules/local/beta_to_OR'

workflow QC_PIPELINE {
    take:
        population  // Population identifier (e.g., "EUR")
        base_dir    // Base directory for the project
    
    main:
        // Define relative paths based on base_dir and population
        raw_dir = "${base_dir}/data/raw/${population}"
        qc_dir = "${base_dir}/data/qc"
        sum_stats_dir = "${base_dir}/data/supplement_data/sum_stats"
        
        // Input files
        input_prefix = "${raw_dir}/${population}"
        qc_prefix = "${qc_dir}/${population}.QC"
        sum_stats = "${sum_stats_dir}/Height.gwas.txt.gz"
        sum_stats_modified = "${sum_stats_dir}/Height.modified.txt.gz"
        sum_stats_qc = "${sum_stats_dir}/Height.QC.gz"
        
        // Step 1: Basic Quality Control
        quality_control(
            input_prefix,
            qc_prefix,
            params.qc.maf ?: 0.01,
            params.qc.mind ?: 0.01,
            params.qc.geno ?: 0.01,
            params.qc.hwe ?: 1e-6
        )

        // Step 1.5: Convert OR to BETA if needed
        or_to_beta(
            sum_stats,
            sum_stats_modified
        )

        // Step 2: Summary Statistics QC
        quality_control_sum(
            or_to_beta.out,
            // sum_stats,
            sum_stats_qc,
            params.qc_sum.info ?: 0.8,
            params.qc_sum.maf ?: 0.01
        )
        
        // Step 3: Pruning
        pruning(
            quality_control.out.collect(),
            input_prefix,
            "${qc_prefix}.fam",
            "${qc_prefix}.snplist",
            qc_prefix
        )
        
        // Step 4: Heterozygosity Check
        heterozygosity(
            pruning.out.collect(),
            input_prefix,
            "${qc_prefix}.fam",
            "${qc_prefix}.prune.in",
            qc_prefix
        )
        
        // Step 5: Identify Valid Samples
        valid_samples(
            heterozygosity.out.collect(),
            "${qc_prefix}.het",
            "${qc_prefix}.valid.sample"
        )
        
        // Step 6: Identify Mismatching SNPs
        mismatching_snps(
            valid_samples.out.collect(),
            "${input_prefix}.bim",
            quality_control_sum.out,
            "${qc_prefix}.snplist",
            "${qc_prefix}.mismatch",
            "${qc_prefix}.a1"
        )
        
        // Step 7: Sex Check
        sex_check(
            valid_samples.out.collect(),
            input_prefix,
            qc_prefix,
            "${qc_prefix}.prune.in",
            "${qc_prefix}.valid.sample",
            "${qc_prefix}.sexcheck",
            "${qc_prefix}.valid"
        )
        
        // Step 8: Relatedness Check
        relatedness(
            sex_check.out.collect(),
            input_prefix,
            qc_prefix,
            "${qc_prefix}.prune.in",
            "${qc_prefix}.valid",
            params.relatedness.cutoff ?: 0.125
        )
        
        // Step 9: QC Wrap-up
        qc_wrap_up(
            relatedness.out.collect(),
            input_prefix,
            qc_prefix,
            "${qc_prefix}.snplist",
            "${qc_prefix}.rel.id",
            "${qc_prefix}.mismatch",
            mismatching_snps.out
        )
        
        // Step 10: Generate Principal Components
        pcs(
            qc_wrap_up.out,
            input_prefix,
            "${qc_prefix}.prune.in",
            params.pcs.pca ?: 6
        )
    
    emit:
        qc_data = qc_wrap_up.out
        pcs = pcs.out
        sum_stats_qc = quality_control_sum.out
}

workflow {
    // Main entry point when run standalone
    QC_PIPELINE(
        params.population ?: "EUR",
        params.base_dir ?: System.getProperty("user.dir")
    )
}