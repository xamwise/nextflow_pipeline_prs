include {quality_control} from '../modules/local/quality_control'
include {quality_control_sum} from '../modules/local/quality_control_sum'
include {pruning} from '../modules/local/pruning'
include {heterozygosity} from '../modules/local/heterozygosity'
include {valid_samples} from '../modules/local/valid_samples'
include {mismatching_snps} from '../modules/local/mismatching_snps'
include {sex_check} from '../modules/local/sex_check'
include {relatedness} from '../modules/local/relatedness'
include {qc_wrap_up} from '../modules/local/qc_wrap_up'
include {pcs} from '../modules/local/pcs'
include {lassosum} from '../modules/local/lassosum'
include {combine_cov} from '../modules/local/combine_cov'
include {prsice} from '../modules/local/prsice'
include {ldpred2} from '../modules/local/ldpred2'
include {prs_cs} from '../modules/local/prs_cs'
include {prs_cs_preprocess} from '../modules/local/prs_cs_preprocess'


workflow {
    quality_control(params.qc.input, params.qc.out, params.qc.maf, params.qc.mind, params.qc.geno, params.qc.hwe)
    quality_control_sum(params.qc_sum.input, params.qc_sum.out, params.qc_sum.info, params.qc_sum.maf)
    pruning(quality_control.out.collect(), params.pruning.input, params.pruning.keep, params.pruning.extract, params.pruning.out)
    heterozygosity(pruning.out.collect(), params.het.input, params.het.keep, params.het.extract, params.het.out)
    valid_samples(heterozygosity.out.collect(), params.valid_samples.input, params.valid_samples.out)
    mismatching_snps(valid_samples.out.collect(), params.mismatching_snps.input, quality_control_sum.out, params.mismatching_snps.snp_list, params.mismatching_snps.mismatch, params.mismatching_snps.out)
    sex_check(valid_samples.out.collect(), params.sex_check.input, params.sex_check.out, params.sex_check.extract, params.sex_check.keep, params.sex_check.update_in, params.sex_check.update_out)
    relatedness(sex_check.out.collect(), params.relatedness.input, params.relatedness.out, params.relatedness.extract, params.relatedness.keep, params.relatedness.cutoff)
    qc_wrap_up(relatedness.out.collect(), params.qc_wrap_up.input, params.qc_wrap_up.out, params.qc_wrap_up.extract, params.qc_wrap_up.keep, params.qc_wrap_up.exclude, mismatching_snps.out)
    pcs(qc_wrap_up.out, params.pcs.out, params.pcs.extract, params.pcs.pca)


    // lassosum(qc_wrap_up.out, params.lassosum.pheno, params.lassosum.cov, pcs.out, params.lassosum.sum_stats, params.lassosum.out)
    // combine_cov(params.combine_cov.cov, pcs.out, params.combine_cov.out)
    // prsice(quality_control_sum.out, params.prsice.pheno, qc_wrap_up.out, combine_cov.out, params.prsice.out, params.prsice.a1, params.prsice.a2, params.prsice.stat, params.prsice.binary_target, params.prsice.base_maf, params.prsice.base_info)
    // ldpred2(qc_wrap_up.out, params.ldpred2.pheno, params.ldpred2.cov, pcs.out, params.ldpred2.ld, quality_control_sum.out, params.ldpred2.trait, params.ldpred2.model, params.ldpred2.out)
    prs_cs_preprocess(quality_control_sum.out, params.prs_cs_preprocess.out)
    prs_cs(params.prs_cs.ref_dir, prs_cs_preprocess.out, params.prs_cs.bim_prefix, params.prs_cs.n_gwas, params.prs_cs.out_dir)
}


