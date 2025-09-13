#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import subworkflows
include { QC_PIPELINE } from './workflows/qc_pipeline'
include { DL_PIPELINE } from './workflows/dl_pipeline'
include { SKLEARN_PIPELINE } from './workflows/sklearn_pipeline'
include { PRS_METHODS } from './workflows/prs_methods'

// Pipeline parameters
params.pipeline_mode = "all"  // Options: all, qc_only, dl_only, sklearn_only, prs_only, ml_comparison
params.run_qc = true
params.run_dl = true
params.run_sklearn = true
params.run_prs = true
params.compare_models = true

// Input/Output parameters
params.input_plink = "${baseDir}/data/genotypes"
params.outdir = "${baseDir}/results"
params.config_dir = "${baseDir}/config"

// Shared parameters
params.seed = 42
params.n_jobs = -1

// Process to prepare data after QC
process PREPARE_ML_DATA {
    publishDir "${params.outdir}/ml_ready", mode: 'copy'
    label 'process_medium'
    
    input:
    path qc_output
    path pheno_file
    
    output:
    path "ml_ready.{bed,bim,fam}", emit: plink_files
    path "phenotypes_clean.csv", emit: phenotypes
    path "data_summary.json", emit: summary
    
    script:
    """
    # Merge QC output with phenotypes
    plink --bfile ${qc_output} \
          --pheno ${pheno_file} \
          --make-bed \
          --out ml_ready
    
    # Clean phenotypes
    python ${baseDir}/scripts/prepare_phenotypes.py \
        --fam ml_ready.fam \
        --pheno ${pheno_file} \
        --output phenotypes_clean.csv \
        --summary data_summary.json
    """
}

// Process to compare ML models (DL vs Sklearn)
process COMPARE_ML_MODELS {
    publishDir "${params.outdir}/model_comparison", mode: 'copy'
    label 'process_high'
    
    input:
    path dl_results
    path sklearn_results
    path prs_results
    
    output:
    path "model_comparison_report.html", emit: report
    path "best_model_selection.json", emit: best_model
    path "comparison_plots.pdf", emit: plots
    path "statistical_tests.csv", emit: stats
    
    script:
    """
    python ${baseDir}/scripts/compare_all_models.py \
        --dl_results ${dl_results} \
        --sklearn_results ${sklearn_results} \
        --prs_results ${prs_results} \
        --output_report model_comparison_report.html \
        --output_selection best_model_selection.json \
        --output_plots comparison_plots.pdf \
        --output_stats statistical_tests.csv
    """
}

// Process to create final ensemble across all methods
process CREATE_SUPER_ENSEMBLE {
    publishDir "${params.outdir}/super_ensemble", mode: 'copy'
    label 'process_high'
    
    input:
    path dl_models
    path sklearn_models
    path prs_scores
    path test_data
    
    output:
    path "super_ensemble_model.pkl", emit: model
    path "ensemble_weights.json", emit: weights
    path "ensemble_performance.json", emit: performance
    
    script:
    """
    python ${baseDir}/scripts/create_super_ensemble.py \
        --dl_models ${dl_models} \
        --sklearn_models ${sklearn_models} \
        --prs_scores ${prs_scores} \
        --test_data ${test_data} \
        --output_model super_ensemble_model.pkl \
        --output_weights ensemble_weights.json \
        --output_performance ensemble_performance.json
    """
}

// Process for final validation on hold-out set
process FINAL_VALIDATION {
    publishDir "${params.outdir}/final_validation", mode: 'copy'
    
    input:
    path best_model
    path holdout_data
    
    output:
    path "final_metrics.json", emit: metrics
    path "final_predictions.csv", emit: predictions
    path "validation_report.html", emit: report
    
    script:
    """
    python ${baseDir}/scripts/final_validation.py \
        --model ${best_model} \
        --data ${holdout_data} \
        --output_metrics final_metrics.json \
        --output_predictions final_predictions.csv \
        --output_report validation_report.html
    """
}

// Main integrated workflow
workflow INTEGRATED_ML_PIPELINE {
    
    // Channel for input data
    input_ch = Channel.fromPath(params.input_plink)
    pheno_ch = Channel.fromPath("${params.input_plink}.pheno")
    
    // Step 1: Quality Control Pipeline
    if (params.run_qc) {
        qc_results = QC_PIPELINE(
            input_ch,
            params.qc
        )
        qc_output = qc_results.final_qc
    } else {
        qc_output = input_ch
    }
    
    // Step 2: Prepare data for ML
    ml_data = PREPARE_ML_DATA(
        qc_output,
        pheno_ch
    )
    
    // Step 3: Run parallel ML pipelines
    ml_results = Channel.empty()
    
    // Deep Learning Pipeline
    if (params.run_dl) {
        dl_results = DL_PIPELINE(
            ml_data.plink_files,
            ml_data.phenotypes,
            params.dl_config
        )
        ml_results = ml_results.mix(dl_results.test_metrics.map { ['dl', it] })
    }
    
    // Sklearn Pipeline
    if (params.run_sklearn) {
        sklearn_results = SKLEARN_PIPELINE(
            ml_data.plink_files,
            ml_data.phenotypes,
            params.sklearn_config
        )
        ml_results = ml_results.mix(sklearn_results.best_metrics.map { ['sklearn', it] })
    }
    
    // PRS Methods Pipeline
    if (params.run_prs) {
        prs_results = PRS_METHODS(
            ml_data.plink_files,
            ml_data.phenotypes,
            params.prs_config
        )
        ml_results = ml_results.mix(prs_results.scores.map { ['prs', it] })
    }
    
    // Step 4: Model Comparison (if multiple pipelines run)
    if (params.compare_models && ml_results.count() > 1) {
        
        // Collect results by type
        dl_metrics = ml_results.filter { it[0] == 'dl' }.map { it[1] }.ifEmpty { Channel.empty() }
        sklearn_metrics = ml_results.filter { it[0] == 'sklearn' }.map { it[1] }.ifEmpty { Channel.empty() }
        prs_metrics = ml_results.filter { it[0] == 'prs' }.map { it[1] }.ifEmpty { Channel.empty() }
        
        // Compare models
        comparison = COMPARE_ML_MODELS(
            dl_metrics.collect().ifEmpty { file("${baseDir}/empty.txt") },
            sklearn_metrics.collect().ifEmpty { file("${baseDir}/empty.txt") },
            prs_metrics.collect().ifEmpty { file("${baseDir}/empty.txt") }
        )
        
        // Create super ensemble if all methods were run
        if (params.run_dl && params.run_sklearn && params.run_prs) {
            super_ensemble = CREATE_SUPER_ENSEMBLE(
                dl_results.models,
                sklearn_results.models,
                prs_results.scores,
                ml_data.plink_files
            )
            
            // Final validation
            final_validation = FINAL_VALIDATION(
                super_ensemble.model,
                ml_data.plink_files
            )
        }
    }
}

// Workflow for sklearn-only pipeline
workflow SKLEARN_ONLY {
    
    input_ch = Channel.fromPath(params.input_plink)
    pheno_ch = Channel.fromPath("${params.input_plink}.pheno")
    
    // Run QC if requested
    if (params.run_qc) {
        qc_results = QC_PIPELINE(input_ch, params.qc)
        processed_data = qc_results.final_qc
    } else {
        processed_data = input_ch
    }
    
    // Prepare ML data
    ml_data = PREPARE_ML_DATA(processed_data, pheno_ch)
    
    // Run sklearn pipeline
    SKLEARN_PIPELINE(
        ml_data.plink_files,
        ml_data.phenotypes,
        params.sklearn_config
    )
}

// Workflow for ML comparison
workflow ML_COMPARISON {
    
    input_ch = Channel.fromPath(params.input_plink)
    pheno_ch = Channel.fromPath("${params.input_plink}.pheno")
    
    // Prepare data
    ml_data = PREPARE_ML_DATA(input_ch, pheno_ch)
    
    // Run both DL and Sklearn pipelines
    dl_results = DL_PIPELINE(
        ml_data.plink_files,
        ml_data.phenotypes,
        params.dl_config
    )
    
    sklearn_results = SKLEARN_PIPELINE(
        ml_data.plink_files,
        ml_data.phenotypes,
        params.sklearn_config
    )
    
    // Compare results
    COMPARE_ML_MODELS(
        dl_results.test_metrics,
        sklearn_results.best_metrics,
        Channel.empty()
    )
}

// Main workflow selector
workflow {
    
    if (params.pipeline_mode == 'all') {
        INTEGRATED_ML_PIPELINE()
    }
    else if (params.pipeline_mode == 'sklearn_only') {
        SKLEARN_ONLY()
    }
    else if (params.pipeline_mode == 'ml_comparison') {
        ML_COMPARISON()
    }
    else if (params.pipeline_mode == 'qc_only') {
        input_ch = Channel.fromPath(params.input_plink)
        QC_PIPELINE(input_ch, params.qc)
    }
    else if (params.pipeline_mode == 'dl_only') {
        input_ch = Channel.fromPath(params.input_plink)
        pheno_ch = Channel.fromPath("${params.input_plink}.pheno")
        DL_PIPELINE(input_ch, pheno_ch, params.dl_config)
    }
    else {
        error "Invalid pipeline_mode: ${params.pipeline_mode}"
    }
}