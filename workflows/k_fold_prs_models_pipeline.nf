#!/usr/bin/env nextflow

/*
 * Polygenic Risk Score (PRS) Models Pipeline
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
include { create_folds } from '../modules/local/create_folds'
include { filter_by_fold } from '../modules/local/filter_by_fold'  // NEW: subsamples inputs to fold train IDs
include { train_evaluate_lr } from '../modules/local/train_evaluate_lr'  // NEW: placeholder — you will implement

// Import QC pipeline if needed
include { QC_PIPELINE } from './qc_pipeline.nf'


/*
 * Sub-workflow: run all enabled PRS models for a single fold.
 *
 * Takes:
 *   fold_id       – integer or string label for this fold (used in output paths)
 *   fold_inputs   – tuple emitted by filter_by_fold:
 *                     [fold_id, train_qc_prefix, train_pheno, train_cov, train_pcs,
 *                      test_ids, train_sum_stats]
 *   base_dirs     – value channel carrying [results_dir, ld_dir, sum_stats_dir, supplement_data_dir]
 */
workflow PRS_FOLD_MODELS {
    take:
        fold_inputs   // channel of: [fold_id, train_qc_prefix, train_pheno, train_cov, train_pcs, test_ids, train_sum_stats]
        base_dirs     // value channel: [results_dir, ld_dir, sum_stats_dir, supplement_data_dir]

    main:

        // Unpack base directories
        base_dirs.map { it }.set { dirs_ch }

        // Derive per-fold output root:  <results_dir>/<population>/fold_<fold_id>/<model>
        fold_results_root = fold_inputs.map { fold_id, qc, pheno, cov, pcs, test_ids, ss ->
            def results_dir = dirs_ch.value[0]
            [fold_id, qc, pheno, cov, pcs, test_ids, ss, "${results_dir}/fold_${fold_id}"]
        }

        // Combine covariates + PCs for this fold
        fold_combine_cov = fold_results_root.map { fold_id, qc, pheno, cov, pcs, test_ids, ss, out_root ->
            [fold_id, cov, pcs, "${out_root}/covariate"]
        }
        combine_cov(fold_combine_cov)

        // Pair combine_cov output back with the rest of the fold data
        fold_with_cov = fold_results_root.join(combine_cov.out, by: 0)
        // fold_with_cov: [fold_id, qc, pheno, cov, pcs, test_ids, ss, out_root, combined_cov]

        // ── Model 1: LassoSum ────────────────────────────────────────────────
        if (params.run_lassosum) {
            lassosum(
                fold_with_cov.map { it[1] },  // qc
                fold_with_cov.map { it[2] },  // pheno
                fold_with_cov.map { it[3] },  // cov
                fold_with_cov.map { it[4] },  // pcs
                fold_with_cov.map { it[6] },  // sum_stats
                fold_with_cov.map { "${it[7]}/lassosum" }
            )
        }

        // ── Model 2: PRSice-2 ────────────────────────────────────────────────
        if (params.run_prsice) {
            prsice(
                fold_with_cov.map { it[6] },  // sum_stats
                fold_with_cov.map { it[2] },  // pheno
                fold_with_cov.map { it[1] },  // qc
                fold_with_cov.map { it[8] },  // combined_cov
                fold_with_cov.map { "${it[7]}/prsice" },
                params.prsice.a1 ?: "A1",
                params.prsice.a2 ?: "A2",
                params.prsice.stat ?: "OR",
                params.prsice.binary_target ?: "F",
                params.prsice.base_maf ?: "MAF:0.01",
                params.prsice.base_info ?: "INFO:0.8"
            )
        }

        // ── Model 3: LDpred2 ─────────────────────────────────────────────────
        if (params.run_ldpred2) {
            ldpred2(
                fold_with_cov.map { it[1] },  // qc
                fold_with_cov.map { it[2] },  // pheno
                fold_with_cov.map { it[3] },  // cov
                fold_with_cov.map { it[4] },  // pcs
                fold_with_cov.map { "${dirs_ch.value[1]}/map.rds" },
                fold_with_cov.map { it[6] },  // sum_stats
                params.ldpred2.trait ?: "quant",
                params.ldpred2.model ?: "inf",
                fold_with_cov.map { "${it[7]}/ldpred2" },
                fold_with_cov.map { it[0].toString() },  // fold_id used as population label
                fold_with_cov.map { it[7] }              // qc_dir equivalent per fold
            )
        }

        // ── Model 4: PRS-CS ──────────────────────────────────────────────────
        if (params.run_prs_cs) {
            prs_cs_preprocess(
                fold_with_cov.map { it[6] },
                fold_with_cov.map { "${dirs_ch.value[2]}/${params.sumstats}.prs_cs.txt" }
            )
            prs_cs(
                fold_with_cov.map { "${dirs_ch.value[1]}/ldblk_1kg_eur" },
                prs_cs_preprocess.out,
                fold_with_cov.map { it[1] },  // qc prefix
                params.prs_cs.n_gwas ?: 20000,
                fold_with_cov.map { "${it[7]}/prs_cs" }
            )
        }

        // ── Model 5: PRS-CSx ─────────────────────────────────────────────────
        if (params.run_prs_csx) {
            prs_csx(
                fold_with_cov.map { dirs_ch.value[1] },
                prs_cs_preprocess.out,
                fold_with_cov.map { it[1] },
                params.prs_csx.n_gwas ?: 20000,
                params.prs_csx.population ?: "EUR",
                fold_with_cov.map { "${it[7]}/prs_csx" },
                "prs_csx"
            )
        }

        // ── Model 6: SBayesR ─────────────────────────────────────────────────
        if (params.run_sbayesr) {
            sbayes_cojo(
                fold_with_cov.map { it[6] },
                fold_with_cov.map { "${dirs_ch.value[2]}/${params.sumstats}.QC.ma" }
            )
            sbayesr(
                sbayes_cojo.out,
                fold_with_cov.map { "${dirs_ch.value[1]}/${params.sbayesr.ld_folder}" },
                "sbayesr_model",
                fold_with_cov.map { "${dirs_ch.value[3]}/${params.sbayesr.annotation}" },
                fold_with_cov.map { it[1] },
                fold_with_cov.map { "${it[7]}/sbayesr" }
            )
        }

        // ── Model 7: PRSet ───────────────────────────────────────────────────
        if (params.run_prset) {
            prset(
                fold_with_cov.map { it[6] },
                fold_with_cov.map { it[2] },
                fold_with_cov.map { it[1] },
                fold_with_cov.map { it[8] },
                fold_with_cov.map { "${it[7]}/prset/prset" },
                params.prset.a1 ?: "A1",
                params.prset.a2 ?: "A2",
                params.prset.stat ?: "OR",
                params.prset.binary_target ?: "F",
                params.prset.base_maf ?: "MAF:0.01",
                params.prset.base_info ?: "INFO:0.8",
                fold_with_cov.map { "${dirs_ch.value[3]}/${params.prset.gtf}" },
                fold_with_cov.map { "${dirs_ch.value[3]}/${params.prset.set}" }
            )
        }

        // ── Model 8: LassoSum2 ───────────────────────────────────────────────
        if (params.run_lassosum2) {
            lassosum2(
                fold_with_cov.map { it[1] },
                fold_with_cov.map { it[2] },
                fold_with_cov.map { it[3] },
                fold_with_cov.map { it[4] },
                fold_with_cov.map { it[6] },
                params.lassosum2.trait,
                params.lassosum2.sample_size,
                fold_with_cov.map { "${it[7]}/lassosum2" }
            )
        }

        // ── Model 9: SCT ─────────────────────────────────────────────────────
        if (params.run_sct) {
            sct(
                fold_with_cov.map { it[1] },
                fold_with_cov.map { it[6] },
                fold_with_cov.map { it[2] },
                params.sct.split ?: 0.7,
                fold_with_cov.map { "${it[7]}/sct/sct" },
                fold_with_cov.map { "${it[7]}/sct" }
            )
        }

    emit:
        // Each emitted channel carries [fold_id, output_path] tuples so the
        // aggregation step below can group results across all folds.
        fold_ids          = fold_inputs.map { it[0] }
        test_ids          = fold_inputs.map { it[5] }   // test IDs file per fold — needed for evaluation
        prsice_results    = params.run_prsice    ? prsice.out    : Channel.empty()
        lassosum_results  = params.run_lassosum  ? lassosum.out  : Channel.empty()
        ldpred2_results   = params.run_ldpred2   ? ldpred2.out   : Channel.empty()
        prs_cs_results    = params.run_prs_cs    ? prs_cs.out    : Channel.empty()
        prs_csx_results   = params.run_prs_csx   ? prs_csx.out   : Channel.empty()
        sbayesr_results   = params.run_sbayesr   ? sbayesr.out   : Channel.empty()
        prset_results     = params.run_prset     ? prset.out     : Channel.empty()
        lassosum2_results = params.run_lassosum2 ? lassosum2.out : Channel.empty()
        sct_results       = params.run_sct       ? sct.out       : Channel.empty()
}


workflow PRS_MODELS {
    take:
        qc_data
        pcs_file
        sum_stats_qc
        population
        base_dir

    main:
        // ── Directory definitions ────────────────────────────────────────────
        raw_dir            = "${base_dir}/data/raw/${population}"
        qc_dir             = "${base_dir}/data/qc"
        results_dir        = "${base_dir}/data/results/${population}"
        ld_dir             = "${base_dir}/data/supplement_data/LD"
        sum_stats_dir      = "${base_dir}/data/supplement_data/sum_stats"
        supplement_data_dir = "${base_dir}/data/supplement_data"

        pheno_file   = "${raw_dir}/${population}.pheno"
        cov_file     = "${raw_dir}/${population}.cov"

        // Bundle static directory paths into a single value channel for
        // easy passage into PRS_FOLD_MODELS without re-specifying each time.
        base_dirs_ch = Channel.value([results_dir, ld_dir, sum_stats_dir, supplement_data_dir])

        // ── 1. Generate cross-validation folds ──────────────────────────────
        // Expected output: channel of [fold_id, train_ids_file, test_ids_file]
        create_folds(
            pheno_file,
            params.folds.n_folds,
            "${qc_dir}/${population}/folds",
            params.folds.random_state
        )

        // ── 2. Subsample all fold-dependent inputs to training IDs ──────────
        //
        // filter_by_fold receives:
        //   [fold_id, train_ids_file, test_ids_file]
        //   + the shared QC plink prefix, pheno, cov, pcs, sum_stats
        //
        // It produces plink --keep filtered genotype files, a filtered pheno,
        // a filtered cov, filtered pcs, and (optionally) filtered sum_stats,
        // returning them as:
        //   [fold_id, train_qc_prefix, train_pheno, train_cov, train_pcs,
        //    test_ids_file, train_sum_stats]
        //
        fold_inputs_ch = filter_by_fold(
            create_folds.out,     // [fold_id, train_ids, test_ids]
            qc_data,
            pheno_file,
            cov_file,
            pcs_file,
            sum_stats_qc,
            "${qc_dir}/${population}/folds"
        )

        // ── 3. Run all PRS models for every fold ─────────────────────────────
        PRS_FOLD_MODELS(fold_inputs_ch, base_dirs_ch)

        // ── 4. Aggregate per-fold results and run LR training / evaluation ───
        //
        // Collect each model's output across ALL folds, then pass the full set
        // to train_evaluate_lr together with the per-fold test-ID lists and the
        // original phenotype file (for true labels).
        //
        // train_evaluate_lr is the module you will implement; it should:
        //   a) Evaluate each model's PRS scores on the held-out test set per fold
        //   b) Concatenate per-fold evaluation metrics
        //   c) Train a stacked LR on the per-fold training predictions
        //   d) Produce final performance summaries
        //
        all_test_ids = PRS_FOLD_MODELS.out.test_ids.collect()

        train_evaluate_lr(
            // Per-fold test ID lists so the script knows which samples to score
            all_test_ids,
            // Raw phenotype file for ground-truth labels
            pheno_file,
            // Collected score files from each model (empty channel if disabled)
            PRS_FOLD_MODELS.out.prsice_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.lassosum_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.ldpred2_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.prs_cs_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.prs_csx_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.sbayesr_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.prset_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.lassosum2_results.collect().ifEmpty([]),
            PRS_FOLD_MODELS.out.sct_results.collect().ifEmpty([]),
            // Output directory for LR artefacts
            "${results_dir}/k_fold_predictive_model_results"
        )

    emit:
        lr_results        = train_evaluate_lr.out
        // Individual model results (all folds combined) still accessible if needed
        prsice_results    = PRS_FOLD_MODELS.out.prsice_results
        lassosum_results  = PRS_FOLD_MODELS.out.lassosum_results
        ldpred2_results   = PRS_FOLD_MODELS.out.ldpred2_results
        prs_cs_results    = PRS_FOLD_MODELS.out.prs_cs_results
        prs_csx_results   = PRS_FOLD_MODELS.out.prs_csx_results
        sbayesr_results   = PRS_FOLD_MODELS.out.sbayesr_results
        prset_results     = PRS_FOLD_MODELS.out.prset_results
        lassosum2_results = PRS_FOLD_MODELS.out.lassosum2_results
        sct_results       = PRS_FOLD_MODELS.out.sct_results
}


workflow {
    if (params.use_existing_qc ?: false) {
        PRS_MODELS(
            Channel.fromPath(params.qc_data_path),
            Channel.fromPath(params.pcs_path),
            Channel.fromPath(params.sum_stats_qc_path),
            params.population ?: "EUR",
            params.base_dir ?: System.getProperty("user.dir")
        )
    } else {
        qc_results = QC_PIPELINE(
            params.population ?: "EUR",
            params.base_dir ?: System.getProperty("user.dir")
        )
        PRS_MODELS(
            qc_results.qc_data.collect(),
            qc_results.pcs,
            qc_results.sum_stats_qc,
            params.population ?: "UKR_CRC",
            params.base_dir ?: System.getProperty("user.dir")
        )
    }
}