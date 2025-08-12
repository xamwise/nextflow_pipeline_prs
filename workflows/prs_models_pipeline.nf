#!/usr/bin/env nextflow

/*
 * Polygenic Risk Score (PRS) Models Pipeline
 * This pipeline applies various PRS methods including:
 * - LassoSum
 * - PRSice-2
 * - LDpred2
 * - PRS-CS
 */

// Import PRS modules
include { lassosum } from '../modules/local/lassosum'
include { combine_cov } from '../modules/local/combine_cov'
include { prsice } from '../modules/local/prsice'
include { ldpred2 } from '../modules/local/ldpred2'
include { prs_cs_preprocess } from '../modules/local/prs_cs_preprocess'
include { prs_cs } from '../modules/local/prs_cs'

// Import QC pipeline if needed
include { QC_PIPELINE } from './qc_pipeline.nf'

workflow PRS_MODELS {
    take:
        qc_data         // QC'd genotype data
        pcs_file        // Principal components file
        sum_stats_qc    // QC'd summary statistics
        population      // Population identifier
        base_dir        // Base directory
    
    main:
        // Define relative paths
        raw_dir = "${base_dir}/data/raw/${population}"
        qc_dir = "${base_dir}/data/qc"
        results_dir = "${base_dir}/data/results/${population}"
        ld_dir = "${base_dir}/data/supplement_data/LD"
        sum_stats_dir = "${base_dir}/data/supplement_data/sum_stats"
        
        // Common input files
        pheno_file = "${raw_dir}/${population}.height"
        cov_file = "${raw_dir}/${population}.cov"
        eigenvec_file = "${raw_dir}/${population}.eigenvec"
        qc_prefix = "${qc_dir}/${population}.QC"
        
        // Combine covariates with PCs
        combine_cov(
            cov_file,
            pcs_file,
            "${raw_dir}/${population}.covariate"
        )
        
        // Model 1: LassoSum
        if (params.run_lassosum ?: true) {
            lassosum(
                qc_data,
                pheno_file,
                cov_file,
                pcs_file,
                sum_stats_qc,
                "${results_dir}/lassosum"
            )
        }
        
        // Model 2: PRSice-2
        if (params.run_prsice ?: true) {
            prsice(
                sum_stats_qc,
                pheno_file,
                qc_data,
                combine_cov.out,
                "${results_dir}/prsice",
                params.prsice.a1 ?: "A1",
                params.prsice.a2 ?: "A2",
                params.prsice.stat ?: "OR",
                params.prsice.binary_target ?: "F",
                params.prsice.base_maf ?: "MAF:0.01",
                params.prsice.base_info ?: "INFO:0.8"
            )
        }
        
        // Model 3: LDpred2
        if (params.run_ldpred2 ?: true) {
            ldpred2(
                qc_data,
                pheno_file,
                cov_file,
                pcs_file,
                "${ld_dir}/map.rds",
                sum_stats_qc,
                params.ldpred2.trait ?: "quant",
                params.ldpred2.model ?: "inf",
                "${results_dir}/ldpred2"
            )
        }
        
        // Model 4: PRS-CS
        if (params.run_prs_cs ?: true) {
            // Preprocess summary statistics for PRS-CS
            prs_cs_preprocess(
                sum_stats_qc,
                "${sum_stats_dir}/Height.prs_cs.txt"
            )
            
            // Run PRS-CS
            prs_cs(
                "${ld_dir}/ldblk_1kg_eur",
                prs_cs_preprocess.out,
                qc_prefix,
                params.prs_cs.n_gwas ?: 20000,
                results_dir
            )
        }
    
    emit:
        lassosum_results = params.run_lassosum ? lassosum.out : Channel.empty()
        prsice_results = params.run_prsice ? prsice.out : Channel.empty()
        ldpred2_results = params.run_ldpred2 ? ldpred2.out : Channel.empty()
        prs_cs_results = params.run_prs_cs ? prs_cs.out : Channel.empty()
}

workflow {
    // Main entry point when run standalone
    // Option 1: Run with pre-existing QC data
    if (params.use_existing_qc ?: false) {
        PRS_MODELS(
            Channel.fromPath(params.qc_data_path),
            Channel.fromPath(params.pcs_path),
            Channel.fromPath(params.sum_stats_qc_path),
            params.population ?: "EUR",
            params.base_dir ?: System.getProperty("user.dir")
        )
    }
    // Option 2: Run QC first, then PRS models
    else {
        qc_results = QC_PIPELINE(
            params.population ?: "EUR",
            params.base_dir ?: System.getProperty("user.dir")
        )
        
        PRS_MODELS(
            qc_results.qc_data,
            qc_results.pcs,
            qc_results.sum_stats_qc,
            params.population ?: "EUR",
            params.base_dir ?: System.getProperty("user.dir")
        )
    }
}