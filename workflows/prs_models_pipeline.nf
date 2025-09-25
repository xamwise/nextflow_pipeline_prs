#!/usr/bin/env nextflow

/*
 * Polygenic Risk Score (PRS) Models Pipeline
 * This pipeline applies various PRS methods to QC'd genetic data and summary statistics including:
 * - LassoSum
 * - PRSice-2
 * - LDpred2
 * - PRS-CS
 * - PRS-CSx
 * - SBAYESR
 * - PRSet
 * - LassoSum2
 * - SCT    
 */

// Import PRS modules
include { lassosum } from '../modules/local/lassosum'
include { combine_cov } from '../modules/local/combine_cov'
include { prsice } from '../modules/local/prsice'
include { ldpred2 } from '../modules/local/ldpred2'
include { prs_cs_preprocess } from '../modules/local/prs_cs_preprocess'
include { prs_cs } from '../modules/local/prs_cs'
include { prs_csx } from '../modules/local/prs_csx'
include { sbayes_cojo } from '../modules/local/sbayes_cojo'
include { sbayesr } from '../modules/local/sbayesr'
include { prset } from '../modules/local/prset'
include { lassosum2 } from '../modules/local/lassosum2'
include { sct } from '../modules/local/sct'

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
        supplement_data_dir = "${base_dir}/data/supplement_data"
        
        // Common input files
        pheno_file = "${raw_dir}/${population}.pheno"
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
        if (params.run_lassosum) {
            lassosum(
                qc_data,
                pheno_file,
                cov_file,
                pcs_file,
                sum_stats_qc,
                "${results_dir}/lassosum/"
            )
        }
        
        // Model 2: PRSice-2
        if (params.run_prsice) {
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
        
        // // Model 3: LDpred2
        if (params.run_ldpred2) {
            ldpred2(
                qc_data,
                pheno_file,
                cov_file,
                pcs_file,
                "${ld_dir}/map.rds",
                sum_stats_qc,
                params.ldpred2.trait ?: "quant",
                params.ldpred2.model ?: "inf",
                "${results_dir}/ldpred2/"
            )
        }
        
        // Model 4: PRS-CS
        if (params.run_prs_cs) {
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
                "${results_dir}/prs_cs/"
            )
        }

        // Model 5: PRS-CSx
        if (params.run_prs_csx) {
            
            // Run PRS-CSx
            prs_csx(
                "${ld_dir}",
                prs_cs_preprocess.out,
                qc_prefix,
                params.prs_csx.n_gwas ?: 20000,
                params.prs_csx.population ?: "EUR",
                "${results_dir}/prs_csx",
                "prs_csx"
            )
        }

        // Model 6: SBAYESR
        if (params.run_sbayesr) {
            // Prepare summary statistics using SBAYES-COJO
            sbayes_cojo(
                sum_stats_qc,
                "${sum_stats_dir}/Height.QC.ma"
            )


            // Run SBAYESR
            sbayesr(
                sbayes_cojo.out,
                "${ld_dir}/${params.sbayesr.ld_folder}",
                "sbayesr_model",
                "${supplement_data_dir}/${params.sbayesr.annotation}",
                qc_prefix,
                "${results_dir}/sbayesr"
            )
        }

        // // Model 7: PRSet 
        if (params.run_prset) {
            prset(
                sum_stats_qc,
                pheno_file,
                qc_data,
                combine_cov.out,
                "${results_dir}/prset/prset",
                params.prset.a1 ?: "A1",
                params.prset.a2 ?: "A2",
                params.prset.stat ?: "OR",
                params.prset.binary_target ?: "F",
                params.prset.base_maf ?: "MAF:0.01",
                params.prset.base_info ?: "INFO:0.8",
                "${supplement_data_dir}/${params.prset.gtf}",
                "${supplement_data_dir}/${params.prset.set}"
            )
        }

        // // Model 8: LassoSum2
        if (params.run_lassosum2) {
            lassosum2(
                qc_data,
                pheno_file,
                cov_file,
                pcs_file,
                sum_stats_qc,
                params.lassosum2.trait,
                params.lassosum2.sample_size,
                "${results_dir}/lassosum2/"
            )
        }

        // Model 9: SCT
        if (params.run_sct) {
            sct(
                qc_data,
                sum_stats_qc,
                pheno_file,
                params.sct.split ?: 0.7,
                "${results_dir}/sct/sct",
                "${results_dir}/sct"
            )
        }

    emit:
        prsice_results = params.run_prsice ? prsice.out : Channel.empty()
        lassosum_results = params.run_lassosum ? lassosum.out : Channel.empty()
        ldpred2_results = params.run_ldpred2 ? ldpred2.out : Channel.empty()
        prs_cs_results = params.run_prs_cs ? prs_cs.out : Channel.empty()
        prs_csx_results = params.run_prs_csx ? prs_csx.out : Channel.empty()
        sbayesr_results = params.run_sbayesr ? sbayesr.out : Channel.empty()
        prset_results = params.run_prset ? prset.out : Channel.empty()
        lassosum2_results = params.run_lassosum2 ? lassosum2.out : Channel.empty()
        sct_results = params.run_sct ? sct.out : Channel.empty()
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