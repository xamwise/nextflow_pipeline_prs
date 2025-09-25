#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Pipeline parameters for sklearn models
//params.input_plink_sklearn = "${params.base_dir}/data/raw/test/test" // PLINK file prefix
params.outdir = "${params.base_dir}/out/sklearn"
params.n_jobs = -1  // Use all available cores

// Default parameters (will be overridden by params file)
// params.data_reuse = [:]
// params.models = [:]
// params.feature_selection = [enabled: false]
// params.splitting = [test_size: 0.2, val_size: 0.1, n_folds: 5, seed: 42]
// params.hyperopt = [enabled: true, n_trials_default: 10]
// params.ensemble = [enabled: false]
// params.interpretability = [shap_analysis: false]

// Process to convert PLINK using DL pipeline's converter
process CONVERT_PLINK_SKLEARN {
    publishDir "${params.outdir}/processed_data", mode: 'copy'
    label 'process_medium'
    
    input:
    path plink_prefix
    
    output:
    path "genotype_data.h5", emit: genotype_data
    path "phenotypes.csv", emit: phenotypes
    path "data_stats.json", emit: stats
    
    script:
    """
    # Use the same converter as DL pipeline
    python ${params.base_dir}/bin/plink_converter.py \
        --plink_prefix ${params.base_dir}${params.input_plink_sklearn} \
        --output_h5 genotype_data.h5 \
        --output_pheno phenotypes.csv \
        --stats_file data_stats.json \
        --phenotype_file ${params.base_dir}${params.input_phenotype_file}
    """
}

// Process for feature selection
process FEATURE_SELECTION {
    publishDir "${params.outdir}/feature_selection", mode: 'copy'
    label 'process_medium'
    
    input:
    path genotype_data
    path phenotypes
    path indices
    
    output:
    path "selected_features.h5", emit: selected_data
    path "feature_importance.csv", emit: importance
    path "selected_indices.npy", emit: indices
    path "feature_selection_report.html", emit: report
    
    when:
    params.feature_selection.enabled
    
    script:
    def methods_str = params.feature_selection.methods.join(',')  // Fix: join array to string
    """
    python ${params.base_dir}/bin/sklearn/feature_selector.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --indices_file ${indices} \
        --n_features ${params.feature_selection.n_features} \
        --methods ${methods_str} \
        --output_data selected_features.h5 \
        --output_importance feature_importance.csv \
        --output_indices selected_indices.npy \
        --output_report feature_selection_report.html \
        --n_jobs ${params.n_jobs}
    """
}

// Process for data splitting using DL pipeline's splitter
process SPLIT_DATA_SKLEARN {
    publishDir "${params.outdir}/data_splits", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    
    output:
    path "splits.json", emit: splits
    path "split_indices.npz", emit: indices
    
    script:
    """
    # Use the same splitter as DL pipeline
    python ${params.base_dir}/bin/data_splitter.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --test_size ${params.splitting.test_size} \
        --val_size ${params.splitting.val_size} \
        --n_folds ${params.splitting.n_folds} \
        --seed ${params.splitting.seed} \
        --output_splits splits.json \
        --output_indices split_indices.npz
    """
}


// Process for hyperparameter optimization
process HYPERPARAMETER_OPTIMIZATION {
    publishDir "${params.outdir}/hyperparameter_optimization/${model_type}", mode: 'copy'
    label 'process_high'
    maxForks 3  // Limit parallel hyperopt jobs
    
    input:
    tuple val(model_type), path(genotype_data), path(phenotypes), path(splits)
    
    output:
    tuple val(model_type), path("best_params_${model_type}.json"), emit: best_params
    path "optuna_study_${model_type}.pkl", emit: study
    path "optimization_history_${model_type}.csv", emit: history
    path "optimization_report_${model_type}.html", emit: report
    
    script:
    def n_trials = params.models[model_type].n_trials ?: params.hyperopt.n_trials_default
    """
    python ${params.base_dir}/bin/sklearn/sklearn_hyperopt.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --cv_folds ${splits} \
        --model_type ${model_type} \
        --n_trials ${n_trials} \
        --n_jobs ${params.n_jobs} \
        --seed ${params.splitting.seed} \
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
          path(splits)
    
    output:
    tuple val(model_type), path("model_${model_type}_fold_*.pkl"), emit: models
    tuple val(model_type), path("cv_results_${model_type}.json"), emit: cv_results
    path "cv_predictions_${model_type}.csv", emit: predictions
    path "training_log_${model_type}.csv", emit: log
    
    script:
    """
    python ${params.base_dir}/bin/sklearn/train_sklearn_cv.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --cv_folds ${splits} \
        --best_params ${best_params} \
        --model_type ${model_type} \
        --n_jobs ${params.n_jobs} \
        --output_models model_${model_type}_fold \
        --output_results cv_results_${model_type}.json \
        --output_predictions cv_predictions_${model_type}.csv \
        --output_log training_log_${model_type}.csv
    """
}

// Process for ensemble model creation (optional)
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
    
    when:
    params.ensemble.enabled
    
    script:
    """
    python ${params.base_dir}/bin/sklearn/create_ensemble.py \
        --models_dir . \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --methods voting,stacking \
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
    def shap_flag = params.interpretability.shap_analysis ? "true" : "false"
    """
    python ${params.base_dir}/bin/sklearn/evaluate_sklearn.py \
        --models ${models} \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --model_type ${model_type} \
        --output_metrics test_metrics_${model_type}.json \
        --output_predictions test_predictions_${model_type}.csv \
        --output_plots evaluation_plots_${model_type}.pdf \
        --output_shap shap_analysis_${model_type}.html \
        --calculate_shap ${shap_flag}
    
    # Create empty SHAP file if not calculated
    if [ "${shap_flag}" = "false" ]; then
        echo "<html><body><p>SHAP analysis disabled</p></body></html>" > shap_analysis_${model_type}.html
    fi
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
    python ${params.base_dir}/bin/sklearn/compare_models.py \
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
    def ensemble_arg = ensemble_performance.name != "none" ? "--ensemble ${ensemble_performance}" : ""
    def features_arg = feature_importance.name != "none" ? "--features ${feature_importance}" : ""
    """
    python ${params.base_dir}/bin/sklearn/generate_report.py \
        --comparison ${comparison_report} \
        ${features_arg} \
        --cv_results ${all_cv_results} \
        ${ensemble_arg} \
        --output_html sklearn_pipeline_report.html \
        --output_pdf sklearn_pipeline_summary.pdf
    """
}

workflow SKLEARN_MODELS {
    take:
        genotype_data
        phenotype_data
        splits_data
        base_dir
        
    main:
        // Feature selection (optional)
        if (params.feature_selection.enabled) {
            FEATURE_SELECTION(
                genotype_data,
                phenotype_data,
                splits_data
            )
            selected_data = FEATURE_SELECTION.out.selected_data
            feature_importance = FEATURE_SELECTION.out.importance
        } else {
            selected_data = genotype_data
            feature_importance = Channel.value(file("none"))
        }
        
        // Prepare channels for enabled models
        models_ch = Channel.empty()
        
        if (params.models.linear.enabled) {
            models_ch = models_ch.mix(Channel.of("linear"))
        }
        if (params.models.ridge.enabled) {
            models_ch = models_ch.mix(Channel.of("ridge"))
        }
        if (params.models.lasso.enabled) {
            models_ch = models_ch.mix(Channel.of("lasso"))
        }
        if (params.models.elasticnet.enabled) {
            models_ch = models_ch.mix(Channel.of("elasticnet"))
        }
        if (params.models.rf.enabled) {
            models_ch = models_ch.mix(Channel.of("rf"))
        }
        if (params.models.gbm.enabled) {
            models_ch = models_ch.mix(Channel.of("gbm"))
        }
        if (params.models.xgboost.enabled) {
            models_ch = models_ch.mix(Channel.of("xgboost"))
        }
        if (params.models.svm.enabled) {
            models_ch = models_ch.mix(Channel.of("svm"))
        }
        
        // Process each model - combine with data
        model_input = models_ch
            .combine(selected_data)
            .combine(phenotype_data)
            .combine(splits_data)
        
        // Create two separate channels for hyperopt and non-hyperopt models
        // Use map to add a flag, then branch based on the flag
        model_with_flag = model_input.map { model_type, geno, pheno, splits ->
            def needs_hyperopt = params.models[model_type].optimize && params.hyperopt.enabled
            tuple(needs_hyperopt, model_type, geno, pheno, splits)
        }
        
        // Split into two channels
        hyperopt_models = model_with_flag
            .filter { it[0] == true }
            .map { flag, model_type, geno, pheno, splits ->
                tuple(model_type, geno, pheno, splits)
            }
        
        non_hyperopt_models = model_with_flag
            .filter { it[0] == false }
            .map { flag, model_type, geno, pheno, splits ->
                tuple(model_type, geno, pheno, splits)
            }
        
        // Run hyperopt for models that need it
        HYPERPARAMETER_OPTIMIZATION(hyperopt_models)
        
        // Prepare training input from hyperopt results
        hyperopt_train_input = HYPERPARAMETER_OPTIMIZATION.out.best_params
            .combine(selected_data)
            .combine(phenotype_data)
            .combine(splits_data)
        
        // Create default params for models that don't need hyperopt
        default_train_input = non_hyperopt_models.map { model_type, geno, pheno, splits ->
            def params_json = params.models[model_type].default_params ?: [:]
            def params_file = file("${params.outdir}/default_params_${model_type}.json")
            params_file.parent.mkdirs()
            params_file.text = groovy.json.JsonOutput.toJson(params_json)
            tuple(model_type, params_file, geno, pheno, splits)
        }
        
        // Combine both training inputs
        all_train_input = hyperopt_train_input.mix(default_train_input)
        
        // Train models
        TRAIN_MODEL_CV(all_train_input)
        
        // Evaluate models
        eval_input = TRAIN_MODEL_CV.out.models
            .combine(selected_data)
            .combine(phenotype_data)
            .combine(splits_data)
        
        EVALUATE_MODEL(eval_input)
        
        // Compare models
        all_metrics = EVALUATE_MODEL.out.metrics.map { _, metrics -> metrics }.collect()
        all_predictions = EVALUATE_MODEL.out.predictions.collect()
        
        COMPARE_MODELS(all_metrics, all_predictions)
        
        // Create ensemble if enabled
        if (params.ensemble.enabled) {
            all_models = TRAIN_MODEL_CV.out.models.map { _, models -> models }.flatten().collect()
            CREATE_ENSEMBLE(
                all_models,
                selected_data,
                phenotype_data,
                splits_data
            )
            ensemble_performance = CREATE_ENSEMBLE.out.performance
        } else {
            ensemble_performance = Channel.value(file("none"))
        }
        
        // Generate report
        all_cv_results = TRAIN_MODEL_CV.out.cv_results.map { _, results -> results }.collect()
        
        GENERATE_REPORT(
            COMPARE_MODELS.out.report,
            feature_importance,
            all_cv_results,
            ensemble_performance
        )
    
    emit:
        models = TRAIN_MODEL_CV.out.models
        metrics = EVALUATE_MODEL.out.metrics
        report = GENERATE_REPORT.out.report
}

workflow {
    // Main entry point
    // Option 1: Use existing converted data
    if (params.data_reuse.use_existing_converted && 
        file(params.data_reuse.existing_genotype_file).exists() &&
        file(params.data_reuse.existing_phenotype_file).exists()) {
        
        genotype_data = Channel.fromPath(params.data_reuse.existing_genotype_file)
        phenotype_data = Channel.fromPath(params.data_reuse.existing_phenotype_file)
    }
    // Option 2: Convert PLINK data
    else {
        CONVERT_PLINK_SKLEARN(
            Channel.fromPath(params.input_plink_sklearn)
        )
        genotype_data = CONVERT_PLINK_SKLEARN.out.genotype_data
        phenotype_data = CONVERT_PLINK_SKLEARN.out.phenotypes
    }
    
    // Option 1: Use existing splits
    if (params.data_reuse.use_existing_splits && 
        file(params.data_reuse.existing_splits_file).exists()) {
        
        splits_data = Channel.fromPath(params.data_reuse.existing_splits_file)
    }
    // Option 2: Create new splits
    else {
        SPLIT_DATA_SKLEARN(
            genotype_data,
            phenotype_data
        )
        splits_data = SPLIT_DATA_SKLEARN.out.indices
    }
    
    
    // Run sklearn models workflow
    SKLEARN_MODELS(
        genotype_data,
        phenotype_data,
        splits_data,
        params.base_dir ?: System.getProperty("user.dir")
    )
}