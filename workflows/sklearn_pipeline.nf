#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Pipeline parameters for sklearn models
params.input_plink = "${baseDir}/data/genotypes" // PLINK file prefix
params.outdir = "${baseDir}/results/sklearn"
params.config = "${baseDir}/workflows/config/sklearn_config.yaml"
params.n_folds = 5
params.test_size = 0.2
params.val_size = 0.1
params.seed = 42
params.n_jobs = -1  // Use all available cores
params.n_trials = 50  // For hyperparameter optimization

// Model types to train
params.models = "linear,ridge,lasso"
params.feature_selection = true
params.n_features = 1000  // Top features to select

// Process to convert PLINK to sklearn-compatible format
process CONVERT_PLINK_SKLEARN {
    publishDir "${params.outdir}/processed_data", mode: 'copy'
    label 'process_medium'
    
    input:
    path plink_prefix
    
    output:
    path "genotype_matrix.npz", emit: genotype_data
    path "phenotypes.csv", emit: phenotypes
    path "feature_names.txt", emit: features
    path "data_stats.json", emit: stats
    
    script:
    """
    python ${baseDir}/scripts/sklearn/plink_to_sklearn.py \
        --plink_prefix ${plink_prefix} \
        --output_matrix genotype_matrix.npz \
        --output_pheno phenotypes.csv \
        --output_features feature_names.txt \
        --stats_file data_stats.json \
        --handle_missing impute
    """
}

// Process for feature selection
process FEATURE_SELECTION {
    publishDir "${params.outdir}/feature_selection", mode: 'copy'
    label 'process_high'
    
    input:
    path genotype_data
    path phenotypes
    path feature_names
    
    output:
    path "selected_features.npz", emit: selected_data
    path "feature_importance.csv", emit: importance
    path "selected_indices.npy", emit: indices
    path "feature_selection_report.html", emit: report
    
    when:
    params.feature_selection
    
    script:
    """
    python ${baseDir}/scripts/sklearn/feature_selector.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --feature_names ${feature_names} \
        --n_features ${params.n_features} \
        --methods univariate,lasso,rf,mutual_info \
        --output_data selected_features.npz \
        --output_importance feature_importance.csv \
        --output_indices selected_indices.npy \
        --output_report feature_selection_report.html \
        --n_jobs ${params.n_jobs}
    """
}

// Process for data splitting
process SPLIT_DATA_SKLEARN {
    publishDir "${params.outdir}/data_splits", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    
    output:
    path "splits.json", emit: splits
    path "split_indices.npz", emit: indices
    path "cv_folds.json", emit: cv_folds
    
    script:
    def input_data = params.feature_selection ? genotype_data : genotype_data
    """
    python ${baseDir}/scripts/sklearn/data_splitter_sklearn.py \
        --genotype_file ${input_data} \
        --phenotype_file ${phenotypes} \
        --test_size ${params.test_size} \
        --val_size ${params.val_size} \
        --n_folds ${params.n_folds} \
        --seed ${params.seed} \
        --stratify auto \
        --output_splits splits.json \
        --output_indices split_indices.npz \
        --output_cv cv_folds.json
    """
}

// Process for hyperparameter optimization
process HYPERPARAMETER_OPTIMIZATION {
    publishDir "${params.outdir}/hyperparameter_optimization/${model_type}", mode: 'copy'
    label 'process_high'
    maxForks 3  // Limit parallel hyperopt jobs
    
    input:
    tuple val(model_type), path(genotype_data), path(phenotypes), path(splits), path(cv_folds)
    
    output:
    tuple val(model_type), path("best_params_${model_type}.json"), emit: best_params
    path "optuna_study_${model_type}.pkl", emit: study
    path "optimization_history_${model_type}.csv", emit: history
    path "optimization_report_${model_type}.html", emit: report
    
    script:
    """
    python ${baseDir}/scripts/sklearn/sklearn_hyperopt.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --cv_folds ${cv_folds} \
        --model_type ${model_type} \
        --n_trials ${params.n_trials} \
        --n_jobs ${params.n_jobs} \
        --seed ${params.seed} \
        --output_params best_params_${model_type}.json \
        --output_study optuna_study_${model_type}.pkl \
        --output_history optimization_history_${model_type}.csv \
        --output_report optimization_report_${model_type}.html
    """
}

// Process for model training with cross-validation
process TRAIN_MODEL_CV {
    publishDir "${params.outdir}/models/${model_type}", mode: 'copy'
    label 'process_high'
    
    input:
    tuple val(model_type), path(best_params), path(genotype_data), path(phenotypes), 
          path(splits), path(cv_folds)
    
    output:
    tuple val(model_type), path("model_${model_type}_fold_*.pkl"), emit: models
    tuple val(model_type), path("cv_results_${model_type}.json"), emit: cv_results
    path "cv_predictions_${model_type}.csv", emit: predictions
    path "training_log_${model_type}.csv", emit: log
    
    script:
    """
    python ${baseDir}/scripts/sklearn/train_sklearn_cv.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --cv_folds ${cv_folds} \
        --best_params ${best_params} \
        --model_type ${model_type} \
        --n_jobs ${params.n_jobs} \
        --output_models model_${model_type}_fold \
        --output_results cv_results_${model_type}.json \
        --output_predictions cv_predictions_${model_type}.csv \
        --output_log training_log_${model_type}.csv
    """
}

// Process for ensemble model creation
process CREATE_ENSEMBLE {
    publishDir "${params.outdir}/ensemble", mode: 'copy'
    label 'process_medium'
    
    input:
    path all_models
    path genotype_data
    path phenotypes
    path splits
    
    output:
    path "ensemble_model.pkl", emit: ensemble
    path "ensemble_weights.json", emit: weights
    path "ensemble_performance.json", emit: performance
    
    script:
    """
    python ${baseDir}/scripts/sklearn/create_ensemble.py \
        --models_dir . \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --methods voting,stacking,blending \
        --output_model ensemble_model.pkl \
        --output_weights ensemble_weights.json \
        --output_performance ensemble_performance.json
    """
}

// Process for model evaluation on test set
process EVALUATE_MODEL {
    publishDir "${params.outdir}/evaluation/${model_type}", mode: 'copy'
    label 'process_medium'
    
    input:
    tuple val(model_type), path(models), path(genotype_data), path(phenotypes), path(splits)
    
    output:
    tuple val(model_type), path("test_metrics_${model_type}.json"), emit: metrics
    path "test_predictions_${model_type}.csv", emit: predictions
    path "evaluation_plots_${model_type}.pdf", emit: plots
    path "shap_analysis_${model_type}.html", emit: shap
    
    script:
    """
    python ${baseDir}/scripts/sklearn/evaluate_sklearn.py \
        --models ${models} \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --model_type ${model_type} \
        --output_metrics test_metrics_${model_type}.json \
        --output_predictions test_predictions_${model_type}.csv \
        --output_plots evaluation_plots_${model_type}.pdf \
        --output_shap shap_analysis_${model_type}.html \
        --calculate_shap true
    """
}

// Process for model comparison
process COMPARE_MODELS {
    publishDir "${params.outdir}/comparison", mode: 'copy'
    
    input:
    path all_metrics
    path all_predictions
    
    output:
    path "model_comparison.html", emit: report
    path "comparison_table.csv", emit: table
    path "comparison_plots.pdf", emit: plots
    path "best_model_summary.json", emit: best_model
    
    script:
    """
    python ${baseDir}/scripts/sklearn/compare_models.py \
        --metrics_files ${all_metrics} \
        --predictions_files ${all_predictions} \
        --output_report model_comparison.html \
        --output_table comparison_table.csv \
        --output_plots comparison_plots.pdf \
        --output_best best_model_summary.json
    """
}

// Process for generating final report
process GENERATE_REPORT {
    publishDir "${params.outdir}", mode: 'copy'
    
    input:
    path comparison_report
    path feature_importance
    path all_cv_results
    path ensemble_performance
    
    output:
    path "sklearn_pipeline_report.html", emit: report
    path "sklearn_pipeline_summary.pdf", emit: summary
    
    script:
    """
    python ${baseDir}/scripts/sklearn/generate_report.py \
        --comparison ${comparison_report} \
        --features ${feature_importance} \
        --cv_results ${all_cv_results} \
        --ensemble ${ensemble_performance} \
        --output_html sklearn_pipeline_report.html \
        --output_pdf sklearn_pipeline_summary.pdf
    """
}

// Main workflow
workflow SKLEARN_PIPELINE {
    
    // Convert PLINK data
    plink_ch = Channel.fromPath(params.input_plink)
    converted = CONVERT_PLINK_SKLEARN(plink_ch)
    
    // Feature selection (optional)
    if (params.feature_selection) {
        selected = FEATURE_SELECTION(
            converted.genotype_data,
            converted.phenotypes,
            converted.features
        )
        genotype_for_training = selected.selected_data
    } else {
        genotype_for_training = converted.genotype_data
    }
    
    // Split data
    splits = SPLIT_DATA_SKLEARN(
        genotype_for_training,
        converted.phenotypes
    )
    
    // Create channel for each model type
    model_types = Channel.from(params.models.tokenize(','))
    
    // Prepare input for hyperparameter optimization
    hyperparam_input = model_types.combine(
        genotype_for_training
    ).combine(
        converted.phenotypes
    ).combine(
        splits.splits
    ).combine(
        splits.cv_folds
    )
    
    // Run hyperparameter optimization
    best_params = HYPERPARAMETER_OPTIMIZATION(hyperparam_input)
    
    // Prepare input for training
    training_input = best_params.best_params.combine(
        genotype_for_training, by: 0
    ).combine(
        converted.phenotypes
    ).combine(
        splits.splits
    ).combine(
        splits.cv_folds
    )
    
    // Train models with cross-validation
    trained_models = TRAIN_MODEL_CV(training_input)
    
    // Create ensemble (collect all trained models)
    all_models = trained_models.models.collect()
    ensemble = CREATE_ENSEMBLE(
        all_models,
        genotype_for_training,
        converted.phenotypes,
        splits.splits
    )
    
    // Evaluate models on test set
    eval_input = trained_models.models.combine(
        genotype_for_training, by: 0
    ).combine(
        converted.phenotypes
    ).combine(
        splits.splits
    )
    
    evaluation = EVALUATE_MODEL(eval_input)
    
    // Compare all models
    all_metrics = evaluation.metrics.collect()
    all_predictions = evaluation.predictions.collect()
    
    comparison = COMPARE_MODELS(
        all_metrics,
        all_predictions
    )
    
    // Generate final report
    all_cv_results = trained_models.cv_results.collect()
    
    if (params.feature_selection) {
        feature_report = selected.importance
    } else {
        feature_report = Channel.empty()
    }
    
    final_report = GENERATE_REPORT(
        comparison.report,
        feature_report,
        all_cv_results,
        ensemble.performance
    )
}

// Workflow entry point
workflow {
    SKLEARN_PIPELINE()
}