#!/usr/bin/env nextflow

/*
 * Main Pipeline Entry Point
 * Orchestrates the complete genetic analysis pipeline:
 * 1. Quality Control
 * 2. PRS Model Application
 * 3. Optional: DL Model Training/Testing
 */

nextflow.enable.dsl = 2

// Import sub-workflows
include { QC_PIPELINE } from './workflows/qc_pipeline.nf'
include { PRS_MODELS } from './workflows/prs_models_pipeline.nf'
// Uncomment when DL pipeline is ready
// include { DL_PIPELINE } from './workflows/dl_pipeline.nf'

// Pipeline version
version = '1.0.0'

// Print pipeline header
log.info """
========================================
 GENETIC ANALYSIS PIPELINE v${version}
========================================
Population      : ${params.population}
Base Directory  : ${params.base_dir}
Run QC          : ${params.run_qc}
Run PRS Models  : ${params.run_prs}
----------------------------------------
"""

workflow {
    // Initialize channels for data flow
    population = params.population ?: "EUR"
    base_dir = params.base_dir ?: System.getProperty("user.dir")
    
    // Step 1: Quality Control Pipeline
    if (params.run_qc ?: true) {
        log.info "Starting Quality Control Pipeline..."
        
        qc_results = QC_PIPELINE(
            population,
            base_dir
        )
        
        qc_data_ch = qc_results.qc_data
        pcs_ch = qc_results.pcs
        sum_stats_qc_ch = qc_results.sum_stats_qc
    } else {
        // Use existing QC results if QC is skipped
        log.info "Skipping QC, using existing QC data..."
        
        qc_dir = "${base_dir}/data/qc"
        raw_dir = "${base_dir}/data/raw/${population}"
        sum_stats_dir = "${base_dir}/data/supplement_data/sum_stats"
        
        qc_data_ch = Channel.fromPath("${qc_dir}/${population}.QC.*")
        pcs_ch = Channel.fromPath("${raw_dir}/${population}.eigenvec")
        sum_stats_qc_ch = Channel.fromPath("${sum_stats_dir}/Height.QC.gz")
    }
    
    // Step 2: PRS Models Pipeline
    if (params.run_prs ?: true) {
        log.info "Starting PRS Models Pipeline..."
        
        prs_results = PRS_MODELS(
            qc_data_ch,
            pcs_ch,
            sum_stats_qc_ch,
            population,
            base_dir
        )
        
        // Log completion of PRS models
        prs_results.lassosum_results.subscribe { 
            log.info "LassoSum completed successfully" 
        }
        prs_results.prsice_results.subscribe { 
            log.info "PRSice-2 completed successfully" 
        }
        prs_results.ldpred2_results.subscribe { 
            log.info "LDpred2 completed successfully" 
        }
        prs_results.prs_cs_results.subscribe { 
            log.info "PRS-CS completed successfully" 
        }
    }
    
    // Step 3: DL Models Pipeline (when implemented)
    if (params.run_dl ?: false) {
        log.info "Starting Deep Learning Pipeline..."
        // DL_PIPELINE(
        //     qc_data_ch,
        //     pcs_ch,
        //     population,
        //     base_dir
        // )
    }
}

// Workflow completion handler
workflow.onComplete {
    log.info """
    ========================================
    Pipeline completed!
    Exit status: ${workflow.exitStatus}
    Duration: ${workflow.duration}
    Success: ${workflow.success}
    ========================================
    """
}