#!/usr/bin/env nextflow

nextflow.enable.dsl=2


// Pipeline parameters - no more hardcoded paths or nextflow.config dependency
// params.base_dir = "/Users/max/Desktop/PRS_Models/nextflow_pipeline_prs"
input_plink = "${params.base_dir}/data/qc/EUR.QC"
outdir = "${params.base_dir}/out"
config = "${params.base_dir}/workflows/config/training_config.yaml"
phenotype_file = "${params.base_dir}/data/raw/A/A.pheno"  // Optional, if phenotype not in PLINK

// Data reuse parameters (lessons from sklearn pipeline)
params.data_reuse = [:]
params.data_reuse.use_existing_converted = false
params.data_reuse.existing_genotype_file = ""
params.data_reuse.existing_phenotype_file = ""
params.data_reuse.use_existing_splits = false
params.data_reuse.existing_splits_file = ""

// Training parameters with defaults
params.n_folds = 5
params.test_size = 0.2
params.val_size = 0.1
params.seed = 42
params.max_epochs = 20
params.batch_size = 500
params.learning_rate = 1e-5

// Make hyperparameter optimization optional
params.hyperopt = [:]
params.hyperopt.enabled = false  // Optional by default
params.hyperopt.n_trials = 20

// Optional wandb
params.use_wandb = false
params.wandb_project = "prs-prediction-DL"
params.wandb_entity = ""

// Process definitions remain mostly the same but with params.base_dir
process CONVERT_PLINK {
    publishDir "${outdir}/processed_data", mode: 'copy'
    
    input:
    path plink_prefix
    
    output:
    path "genotype_data.h5", emit: genotype_data
    path "phenotypes.csv", emit: phenotypes
    path "data_stats.json", emit: stats
    
    script:
    """
    python ${params.base_dir}/bin/plink_converter.py \
        --plink_prefix ${input_plink} \
        --output_h5 genotype_data.h5 \
        --output_pheno phenotypes.csv \
        --stats_file data_stats.json \
        --phenotype_file ${phenotype_file}
    """
}

process SPLIT_DATA {
    publishDir "${outdir}/data_splits", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    
    output:
    path "splits.json", emit: splits
    path "split_indices.npz", emit: indices
    
    script:
    """
    python ${params.base_dir}/bin/data_splitter.py \
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

process VISUALIZE_DATA {
    publishDir "${outdir}/visualizations", mode: 'copy'
    
    input:
    path genotype_data
    path phenotypes
    path stats
    
    output:
    path "*.png", optional: true
    path "*.html", optional: true
    path "visualization_report.pdf", optional: true
    
    when:
    params.create_visualizations
    
    script:
    """
    python ${params.base_dir}/bin/data_visualizer.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --stats_file ${stats} \
        --output_dir .
    """
}

process HYPERPARAMETER_OPTIMIZATION {
    publishDir "${outdir}/hyperparameter_search", mode: 'copy'
    
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
    
    when:
    params.hyperopt.enabled
    
    script:
    """
    python ${params.base_dir}/bin/hyperparameter_optimizer.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --splits_file ${splits} \
        --indices_file ${indices} \
        --config ${config} \
        --n_trials ${params.hyperopt.n_trials} \
        --output_params best_params.yaml \
        --study_db optuna_study.db \
        --report optimization_report.html
    """
}

process TRAIN_KFOLD {
    tag "fold_${fold}"
    publishDir "${outdir}/models/fold_${fold}", mode: 'copy'
    
    input:
    tuple val(fold), path(genotype_data), path(phenotypes), path(indices), path(params_file)
    
    output:
    tuple val(fold), path("model_fold_${fold}.pt"), emit: model
    path "metrics_fold_${fold}.json", emit: metrics
    path "training_log_fold_${fold}.csv", emit: log
    
    script:
    def wandb_args = params.use_wandb ? 
        "--wandb_project ${params.wandb_project} --wandb_entity ${params.wandb_entity}" : ""
    """
    python ${params.base_dir}/bin/train_model.py \
        --genotype_file ${genotype_data} \
        --phenotype_file ${phenotypes} \
        --indices_file ${indices} \
        --params_file ${params_file} \
        --fold ${fold} \
        --max_epochs ${params.max_epochs} \
        --batch_size ${params.batch_size} \
        --output_model model_fold_${fold}.pt \
        --output_metrics metrics_fold_${fold}.json \
        --output_log training_log_fold_${fold}.csv \
        ${wandb_args}
    """
}

process EVALUATE_MODELS {
    publishDir "${outdir}/evaluation", mode: 'copy'
    
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
    def model_list = models.collect{ it.name }.join(' ')
    def metrics_list = metrics.collect{ it.name }.join(' ')
    """
    python ${params.base_dir}/bin/evaluate_models.py \
        --models ${model_list} \
        --metrics ${metrics_list} \
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
    config_ch = Channel.fromPath(config)
    
    // Step 1: Convert PLINK or use existing converted data
    if (params.data_reuse.use_existing_converted && 
        file(params.data_reuse.existing_genotype_file).exists() &&
        file(params.data_reuse.existing_phenotype_file).exists()) {
        
        genotype_data = Channel.fromPath(params.data_reuse.existing_genotype_file)
        phenotype_data = Channel.fromPath(params.data_reuse.existing_phenotype_file)
        stats_data = Channel.empty()  // Stats might not exist for reused data
        
    } else {
        plink_ch = Channel.fromPath(input_plink)
        CONVERT_PLINK(plink_ch)
        genotype_data = CONVERT_PLINK.out.genotype_data
        phenotype_data = CONVERT_PLINK.out.phenotypes
        stats_data = CONVERT_PLINK.out.stats
    }
    
    // Step 2: Split data or use existing splits
    if (params.data_reuse.use_existing_splits && 
        file(params.data_reuse.existing_splits_file).exists()) {
        
        splits_json = Channel.empty()  // May not be needed
        splits_indices = Channel.fromPath(params.data_reuse.existing_splits_file)
        
    } else {
        SPLIT_DATA(genotype_data, phenotype_data)
        splits_json = SPLIT_DATA.out.splits
        splits_indices = SPLIT_DATA.out.indices
    }
    
    // Step 3: Optional visualization
    if (params.create_visualizations && !params.data_reuse.use_existing_converted) {
        VISUALIZE_DATA(genotype_data, phenotype_data, stats_data)
    }
    
    // Step 4: Determine parameters to use (hyperopt or config)
    if (params.hyperopt.enabled) {
        HYPERPARAMETER_OPTIMIZATION(
            genotype_data,
            phenotype_data,
            splits_json.ifEmpty(Channel.value(file("dummy_splits.json"))),
            splits_indices,
            config_ch
        )
        params_to_use = HYPERPARAMETER_OPTIMIZATION.out.best_params
    } else {
        params_to_use = config_ch
    }
    
    // Step 5: Train models with k-fold cross-validation
    // Fix: Create fold channel properly
    fold_ch = Channel.from(0..(params.n_folds-1))
    
    train_input = fold_ch
        .combine(genotype_data)
        .combine(phenotype_data)
        .combine(splits_indices)
        .combine(params_to_use)
    
    TRAIN_KFOLD(train_input)
    
    // Step 6: Collect and evaluate
    all_models = TRAIN_KFOLD.out.model.map { fold, model -> model }.collect()
    all_metrics = TRAIN_KFOLD.out.metrics.collect()
    
    EVALUATE_MODELS(
        all_models,
        all_metrics,
        genotype_data,
        phenotype_data,
        splits_indices
    )
}

workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    println "Results saved to: ${outdir}"
}