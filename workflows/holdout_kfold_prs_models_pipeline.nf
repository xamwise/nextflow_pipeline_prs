#!/usr/bin/env nextflow

/*
 * Polygenic Risk Score (PRS) Models Pipeline with hold-out test set and k-fold cross-validation
 *
 * Based on prs_models_pipeline.nf, but the PRS models are not trained on all samples:
 *   1. The .pheno file is aligned to the QC'd .fam (one row per .fam sample, in .fam order).
 *   2. Hold-out split: holdout.test_size (default 10%) of the samples are set aside as hold-out
 *      test set, the remaining samples form the development set.
 *   3. K-fold split of the development set into folds.n_folds folds (create_folds).
 *   4. For every fold i the training set (development set without fold i) and the validation set
 *      (fold i) are written as PLINK files plus .pheno / .cov / .eigenvec / .covariate files.
 *      The hold-out test set is written the same way.
 *   5. All enabled PRS models are trained on the training set of every fold.
 *
 * The validation and hold-out test sets are not used by the pipeline. They are scored afterwards
 * with the SNP weights of the fold models in collect_results_prs_models_kfold.ipynb.
 *
 * Output layout (<pop> = population):
 *   data/qc/<pop>/kfold/
 *       <pop>.pheno                                    .pheno aligned to the QC'd .fam
 *       holdout/test_ids.txt, dev_ids.txt, dev.pheno  hold-out split
 *       folds/fold_<i>.txt                             validation IDs of fold i
 *       test/<pop>.QC.{bed,bim,fam}, <pop>.pheno, <pop>.cov, <pop>.eigenvec, <pop>.covariate
 *       fold_<i>/train/...                             same files as in test/
 *       fold_<i>/val/...                               same files as in test/
 *   data/results/<pop>/kfold/fold_<i>/<model>/        models trained on fold_<i>/train
 *
 * SNP weights per fold model:
 *   lassosum/_betas.csv, lassosum2/lassosum2_betas.csv, ldpred2/ldpred2_betas.csv,
 *   ldpred2_cli/ldpred2_cli.<auto|inf>_betas.csv, sct/sct_betas.csv, sbayesr/sbayesr_model_sbrc.txt,
 *   sbayesrc_gctb/sbayesrc.snpRes, prs_cs/*_chr<chr>.txt, prs_csx/*_chr<chr>.txt,
 *   prsice/prsice.snp (best threshold in prsice/prsice.summary), prset/prset.snp
 *
 * Usage:
 *   nextflow run workflows/holdout_kfold_prs_models_pipeline.nf -params-file workflows/config/params_prs_kfold.yaml
 */

// Import PRS modules
include { lassosum } from '../modules/local/lassosum'
include { combine_cov } from '../modules/local/combine_cov'
include { prsice_kfold } from '../modules/local/prsice_kfold'
include { ldpred2 } from '../modules/local/ldpred2'
include { prs_cs_preprocess } from '../modules/local/prs_cs_preprocess'
include { prs_cs } from '../modules/local/prs_cs'
include { prs_csx } from '../modules/local/prs_csx'
include { sbayes_cojo } from '../modules/local/sbayes_cojo'
include { sbayesr } from '../modules/local/sbayesr'
include { prset_kfold } from '../modules/local/prset_kfold'
include { lassosum2 } from '../modules/local/lassosum2'
include { sct } from '../modules/local/sct'
include { ldpred2_cli_kfold } from '../modules/local/ldpred2_cli_kfold'
include { genetic_maps } from '../modules/local/genetic_maps'
include { gctb_ma } from '../modules/local/gctb_ma'
include { sbayesrc_gctb } from '../modules/local/sbayesrc_gctb'

// Import data preparation and splitting modules
include { align_pheno } from '../modules/local/align_pheno'
include { holdout_split } from '../modules/local/holdout_split'
include { create_folds } from '../modules/local/create_folds'
include { subset_samples } from '../modules/local/subset_samples'

workflow KFOLD_PRS_MODELS {
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
        ld_dir = "${base_dir}/data/supplement_data/LD"
        sum_stats_dir = "${base_dir}/data/supplement_data/sum_stats"
        supplement_data_dir = "${base_dir}/data/supplement_data"

        // Splits and fold models go to separate 'kfold' folders, so the outputs of
        // prs_models_pipeline.nf are not overwritten
        kfold_dir = "${qc_dir}/${population}/kfold"
        kfold_results_dir = "${base_dir}/data/results/${population}/kfold"

        // Common input files
        pheno_file = "${raw_dir}/${population}.pheno"
        cov_file = "${raw_dir}/${population}.cov"

        n_folds = params.folds.n_folds
        test_size = params.holdout?.test_size ?: 0.1
        holdout_random_state = params.holdout?.random_state != null ? params.holdout.random_state : params.folds.random_state

        // Value channels, so these inputs are used for every split and fold
        qc_prefix = qc_data.first()
        pcs_path = pcs_file.first()
        sum_stats = sum_stats_qc.first()

        // ── 1. Align the .pheno file to the QC'd .fam ─────────────────────────
        // Writes <kfold_dir>/<pop>.pheno: one row per .fam sample in .fam order, NA if no phenotype
        align_pheno(
            pheno_file,
            qc_prefix.map { "${it}.fam" },
            "${kfold_dir}/${population}.pheno"
        )

        // ── 2. Hold-out split ────────────────────────────────────────────────
        // Writes <kfold_dir>/holdout/{test_ids.txt, dev_ids.txt, dev.pheno}
        holdout_split(
            align_pheno.out,
            qc_prefix.map { "${it}.fam" },
            test_size,
            "${kfold_dir}/holdout",
            holdout_random_state
        )

        // ── 3. K-fold split of the development set ───────────────────────────
        // Writes <kfold_dir>/folds/fold_<i>.txt with the validation IDs of fold i
        create_folds(
            holdout_split.out.dev_pheno,
            n_folds,
            kfold_dir,
            params.folds.random_state
        )

        // Combine covariates with PCs (waits for align_pheno, which creates the kfold folder)
        combine_cov(
            cov_file,
            pcs_path,
            align_pheno.out.map { "${kfold_dir}/${population}.covariate" }
        )

        // ── 4. Write genotypes and sample files of every split ───────────────
        // Items: [fold_id, split, out_dir, keep_ids, remove_ids]
        test_split = holdout_split.out.test_ids.map { test_ids ->
            ['holdout', 'test', "${kfold_dir}/test", test_ids, 'NONE']
        }

        fold_splits = create_folds.out.flatMap {
            (1..n_folds).collectMany { i ->
                def val_ids = "${kfold_dir}/folds/fold_${i}.txt"
                [
                    // Training set: development set without fold i
                    [i, 'train', "${kfold_dir}/fold_${i}/train", "${kfold_dir}/holdout/dev_ids.txt", val_ids],
                    // Validation set: fold i
                    [i, 'val', "${kfold_dir}/fold_${i}/val", val_ids, 'NONE']
                ]
            }
        }

        subset_samples(
            test_split.mix(fold_splits),
            qc_prefix,
            align_pheno.out,
            cov_file,
            pcs_path,
            combine_cov.out,
            population
        )

        // ── 5. Train all PRS models on the training set of every fold ────────
        // multiMap keeps the per-fold inputs of each model call aligned
        train = subset_samples.out
            .filter { it[1] == 'train' }
            .multiMap { fold_id, split, bed, pheno, cov, eigenvec, covariate, data_dir ->
                bed: bed
                pheno: pheno
                cov: cov
                pcs: eigenvec
                covariate: covariate
                data_dir: data_dir
                results: "${kfold_results_dir}/fold_${fold_id}"
            }

        // Genetic maps for the LD matrices of LDpred2 / LassoSum2, downloaded once into a shared folder
        // (downloads by every fold in parallel time out)
        if (params.run_ldpred2 || params.run_lassosum2) {
            genetic_maps("${supplement_data_dir}/genetic_maps")
        }

        // Model 1: LassoSum
        if (params.run_lassosum) {
            lassosum(
                train.bed,
                train.pheno,
                train.cov,
                train.pcs,
                sum_stats,
                train.results.map { "${it}/lassosum/" }
            )
        }

        // Model 2: PRSice-2
        if (params.run_prsice) {
            prsice_kfold(
                sum_stats,
                train.pheno,
                train.bed,
                train.covariate,
                train.results.map { "${it}/prsice" },
                params.prsice.a1 ?: "A1",
                params.prsice.a2 ?: "A2",
                params.prsice.stat ?: "OR",
                params.prsice.binary_target ?: "F",
                params.prsice.base_maf ?: "MAF:0.01",
                params.prsice.base_info ?: "INFO:0.8"
            )
        }

        // Model 3: LDpred2
        // LDpred-2.R keeps its temp files in <data_dir>/<population>/tmp-data (derived from the bed prefix)
        if (params.run_ldpred2) {
            ldpred2(
                train.bed,
                train.pheno,
                train.cov,
                train.pcs,
                "${ld_dir}/map.rds",
                sum_stats,
                params.ldpred2.trait ?: "quant",
                params.ldpred2.model ?: "inf",
                train.results.map { "${it}/ldpred2/" },
                population,
                train.data_dir,
                genetic_maps.out
            )
        }

        if (params.run_ldpred2_cli) {
            ldpred2_cli_kfold(
                train.bed,
                train.pheno,
                train.cov,
                train.pcs,
                "${ld_dir}/map.rds",
                sum_stats,
                params.ldpred2.trait ?: "quant",
                params.ldpred2.model ?: "auto",
                train.results.map { "${it}/ldpred2_cli/" },
                population,
                train.data_dir
            )
        }

        // Preprocess summary statistics for PRS-CS / PRS-CSx
        // (does not depend on the fold and writes to a shared file, so it only runs once)
        if (params.run_prs_cs || params.run_prs_csx) {
            prs_cs_preprocess(
                sum_stats,
                "${sum_stats_dir}/${params.sumstats}.prs_cs.txt"
            )
        }

        // Model 4: PRS-CS
        if (params.run_prs_cs) {
            prs_cs(
                "${ld_dir}/ldblk_1kg_eur",
                prs_cs_preprocess.out,
                train.bed,
                params.prs_cs.n_gwas ?: 20000,
                train.results.map { "${it}/prs_cs/" }
            )
        }

        // Model 5: PRS-CSx
        if (params.run_prs_csx) {
            prs_csx(
                "${ld_dir}",
                prs_cs_preprocess.out,
                train.bed,
                params.prs_csx.n_gwas ?: 20000,
                params.prs_csx.population ?: "EUR",
                train.results.map { "${it}/prs_csx" },
                "prs_csx"
            )
        }

        // Model 6: SBAYESR
        if (params.run_sbayesr) {
            // Prepare summary statistics using SBAYES-COJO
            // (does not depend on the fold and writes to a shared file, so it only runs once)
            sbayes_cojo(
                sum_stats,
                "${sum_stats_dir}/${params.sumstats}.QC.ma"
            )

            // Run SBAYESR
            sbayesr(
                train.bed,
                sbayes_cojo.out,
                "${ld_dir}/${params.sbayesr.ld_folder}",
                "sbayesr_model",
                "${supplement_data_dir}/${params.sbayesr.annotation}",
                train.bed,
                train.results.map { "${it}/sbayesr" }
            )
        }

        // Model 6b: SBayesRC with the GCTB command line tool (second SBayesRC implementation)
        if (params.run_sbayesrc_gctb) {
            gctb_ld = "${ld_dir}/${params.sbayesrc_gctb?.ld_folder ?: params.sbayesr?.ld_folder}"

            // Summary statistics in the GCTB .ma format (log(OR) for binary traits, frequency of A1)
            // (does not depend on the fold and writes to a shared file, so it only runs once)
            gctb_ma(
                sum_stats,
                "${gctb_ld}/snp.info",
                "${sum_stats_dir}/${params.sumstats}.gctb.ma"
            )

            sbayesrc_gctb(
                gctb_ma.out,
                gctb_ld,
                "${supplement_data_dir}/${params.sbayesrc_gctb?.annotation ?: params.sbayesr?.annotation}",
                train.bed,
                train.results.map { "${it}/sbayesrc_gctb" }
            )
        }

        // Model 7: PRSet
        if (params.run_prset) {
            prset_kfold(
                sum_stats,
                train.pheno,
                train.bed,
                train.covariate,
                train.results.map { "${it}/prset/prset" },
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

        // Model 8: LassoSum2
        if (params.run_lassosum2) {
            lassosum2(
                train.bed,
                train.pheno,
                train.cov,
                train.pcs,
                sum_stats,
                params.lassosum2.trait,
                params.lassosum2.sample_size,
                train.results.map { "${it}/lassosum2/" },
                genetic_maps.out
            )
        }

        // Model 9: SCT
        if (params.run_sct) {
            sct(
                train.bed,
                sum_stats,
                train.pheno,
                params.sct.split ?: 0.7,
                train.results.map { "${it}/sct/sct" },
                train.results.map { "${it}/sct" }
            )
        }

    emit:
        // [fold_id, split, bed_prefix, pheno, cov, eigenvec, covariate, dir] for the train / val / test sets
        split_data = subset_samples.out
        prsice_results = params.run_prsice ? prsice_kfold.out : Channel.empty()
        lassosum_results = params.run_lassosum ? lassosum.out : Channel.empty()
        ldpred2_results = params.run_ldpred2 ? ldpred2.out : Channel.empty()
        ldpred2_cli_results = params.run_ldpred2_cli ? ldpred2_cli_kfold.out : Channel.empty()
        prs_cs_results = params.run_prs_cs ? prs_cs.out : Channel.empty()
        prs_csx_results = params.run_prs_csx ? prs_csx.out : Channel.empty()
        sbayesr_results = params.run_sbayesr ? sbayesr.out : Channel.empty()
        sbayesrc_gctb_results = params.run_sbayesrc_gctb ? sbayesrc_gctb.out : Channel.empty()
        prset_results = params.run_prset ? prset_kfold.out : Channel.empty()
        lassosum2_results = params.run_lassosum2 ? lassosum2.out : Channel.empty()
        sct_results = params.run_sct ? sct.out : Channel.empty()
}

workflow {
    // Runs on existing QC data only (run workflows/qc_pipeline.nf first, see run_pipelines.sh):
    // QC_PIPELINE does not emit the PCs, so it cannot be chained in front of this pipeline.
    if (!params.use_existing_qc) {
        error "holdout_kfold_prs_models_pipeline.nf needs existing QC data: run workflows/qc_pipeline.nf first and set use_existing_qc: true"
    }

    KFOLD_PRS_MODELS(
        Channel.fromPath(params.qc_data_path),
        Channel.fromPath(params.pcs_path),
        Channel.fromPath(params.sum_stats_qc_path),
        params.population ?: "EUR",
        params.base_dir ?: System.getProperty("user.dir")
    )
}
