#!/usr/bin/env nextflow

nextflow.enable.dsl=2

base_dir = "/Users/max/Desktop/PRS_Models/nextflow_pipeline_prs"

// Pipeline parameters for Deep Learning-based Polygenic Risk Score Prediction
params.input_plink = "${base_dir}/data/qc" // PLINK file prefix
params.outdir = "${base_dir}/out"
params.config = "${base_dir}/workflows/config/training_config.yaml"
params.n_folds = 5
params.test_size = 0.2
params.val_size = 0.1
params.seed = 42
params.max_epochs = 20
params.batch_size = 500
params.learning_rate = 1e-5
params.n_trials = 20 // for hyperparameter optimization
params.wandb_project = "prs-prediction-DL"
params.wandb_entity = "your-entity" // Replace with your W&B entity

// Process to convert PLINK to PyTorch-compatible format
process CONVERT_PLINK {
    publishDir "${params.outdir}/processed_data", mode: 'copy'
    
    input:
    path plink_prefix
    
    output:
    path "genotype_data.h5", emit: genotype_data
    path "phenotypes.csv", emit: phenotypes
    path "data_stats.json", emit: stats
    
    script:
    """
    python ${base_dir}/bin/plink_converter.py \
        --plink_prefix ${plink_prefix} \
        --output_h5 genotype_data.h5 \
        --output_pheno phenotypes.csv \
        --stats_file data_stats.json
    """
}

// Process for data splitting
process SPLIT_DATA {
    publishDir "${params.outdir}/data_splits", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    
    output:
    path "splits.json", emit: splits
    path "split_indices.npz", emit: indices
    
    script:
    """
    python ${base_dir}/bin/data_splitter.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --test_size ${params.test_size} \
        --val_size ${params.val_size} \
        --n_folds ${params.n_folds} \
        --seed ${params.seed} \
        --output_splits splits.json \
        --output_indices split_indices.npz
    """
}

// Process for data visualization
process VISUALIZE_DATA {
    publishDir "${params.outdir}/visualizations", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    path stats
    
    output:
    path "*.png"
    path "*.html"
    path "visualization_report.pdf"
    
    script:
    """
    python ${base_dir}/bin/data_visualizer.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --stats_file ${stats} \
        --output_dir .
    """
}

// Process for hyperparameter optimization
process HYPERPARAMETER_OPTIMIZATION {
    publishDir "${params.outdir}/hyperparameter_search", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    path splits
    path indices
    path config
    
    output:
    path "best_params.yaml", emit: best_params
    path "optuna_study.db", emit: study_db
    path "optimization_report.html", emit: report
    
    script:
    """
    python ${base_dir}/bin/hyperparameter_optimizer.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --indices_file ${indices} \
        --config ${config} \
        --n_trials ${params.n_trials} \
        --output_params best_params.yaml \
        --study_db optuna_study.db \
        --report optimization_report.html
    """
}

// Process for k-fold cross-validation training
process TRAIN_KFOLD {
    tag "fold_${fold}"
    publishDir "${params.outdir}/models/fold_${fold}", mode: 'copy'
    
    input:
    tuple val(fold), path(genotype_data), path(phenotypes), path(indices), path(best_params)
    
    output:
    path "model_fold_${fold}.pt", emit: model
    path "metrics_fold_${fold}.json", emit: metrics
    path "training_log_fold_${fold}.csv", emit: log
    
    script:
    """
    python ${base_dir}/bin/train_model.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --indices_file ${indices} \
        --params_file ${best_params} \
        --fold ${fold} \
        --max_epochs ${params.max_epochs} \
        --batch_size ${params.batch_size} \
        --output_model model_fold_${fold}.pt \
        --output_metrics metrics_fold_${fold}.json \
        --output_log training_log_fold_${fold}.csv \
        --wandb_project ${params.wandb_project} \
        --wandb_entity ${params.wandb_entity}
    """
}

// Process for final model evaluation
process EVALUATE_MODELS {
    publishDir "${params.outdir}/evaluation", mode: 'copy'
    
    input:
    path models
    path metrics
    path genotype_data
    path phenotypes
    path indices
    
    output:
    path "final_evaluation_report.html"
    path "ensemble_predictions.csv"
    path "model_comparison.json"
    
    script:
    """
    python ${base_dir}/bin/evaluate_models.py \
        --models ${models} \
        --metrics ${metrics} \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --indices_file ${indices} \
        --output_report final_evaluation_report.html \
        --output_predictions ensemble_predictions.csv \
        --output_comparison model_comparison.json
    """
}

// Main workflow
workflow {
    // Input channel
    plink_ch = Channel.fromPath(params.input_plink)
    config_ch = Channel.fromPath(params.config)
    
    // Convert PLINK files
    CONVERT_PLINK(plink_ch)
    
    // Split data
    SPLIT_DATA(
        CONVERT_PLINK.out.genotype_data,
        CONVERT_PLINK.out.phenotypes
    )
    
    // Visualize data
    VISUALIZE_DATA(
        CONVERT_PLINK.out.genotype_data,
        CONVERT_PLINK.out.phenotypes,
        CONVERT_PLINK.out.stats
    )
    
    // Optimize hyperparameters
    HYPERPARAMETER_OPTIMIZATION(
        CONVERT_PLINK.out.genotype_data,
        CONVERT_PLINK.out.phenotypes,
        SPLIT_DATA.out.splits,
        SPLIT_DATA.out.indices,
        config_ch
    )
    
    // Create fold channel for k-fold training
    fold_ch = Channel.from(0..params.n_folds-1)
        .combine(Channel.value(CONVERT_PLINK.out.genotype_data))
        .combine(Channel.value(CONVERT_PLINK.out.phenotypes))
        .combine(Channel.value(SPLIT_DATA.out.indices))
        .combine(Channel.value(HYPERPARAMETER_OPTIMIZATION.out.best_params))
    
    // Train models with k-fold cross-validation
    TRAIN_KFOLD(fold_ch)
    
    // Collect all models and metrics
    all_models = TRAIN_KFOLD.out.model.collect()
    all_metrics = TRAIN_KFOLD.out.metrics.collect()
    
    // Evaluate all models
    EVALUATE_MODELS(
        all_models,
        all_metrics,
        CONVERT_PLINK.out.genotype_data,
        CONVERT_PLINK.out.phenotypes,
        SPLIT_DATA.out.indices
    )
}

workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    println "Results saved to: ${params.outdir}"
}