/*
 * PRS-Net feature construction module.
 *
 * Emits the same three artifacts as CONVERT_PLINK (genotype_data / phenotypes /
 * stats), so the downstream SPLIT_DATA -> TRAIN_KFOLD -> EVALUATE_MODELS chain in
 * dl_prs.nf is untouched. The only difference is what lives inside the h5:
 * (N, n_genes, 11) gene-level PRS instead of (N, n_snps[, c]) genotypes.
 *
 * Include from dl_prs.nf with:
 *     include { PRSNET_FEATURES } from './workflows/modules/prsnet.nf'
 */

nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
// Parameter resolution
// ---------------------------------------------------------------------------
// params come from `-params-file workflows/config/dl_config.yaml`, so the
// `prsnet` map may be absent entirely (a non-PRS-Net run) or only partially
// filled in. Two things to know:
//
//   1. A nested map supplied via -params-file REPLACES a script-level default
//      map wholesale -- it does not merge key by key. So defaults must be
//      merged explicitly, as below, or half the keys silently become null.
//   2. `${...}` interpolation does NOT work inside a YAML params file. Paths in
//      dl_config.yaml must be literal, not "${params.base_dir}/...".
//
// prsnetParam() merges user values over these defaults and fails with a usable
// message rather than "Cannot get property 'x' on null object".

def PRSNET_DEFAULTS = [
    genes_per_chunk : 250,
    clump_r2        : 0.5,
    clump_kb        : 250,
    min_maf         : 0.001,
    min_info        : 0.3,
    target_maf      : 0.01,
    target_hwe      : 1e-6,
    target_geno     : 0.01,
    target_mind     : 0.01,
    plink_memory    : 16000,
    plink_bin       : 'plink',
    plink2_bin      : './bin/plink2',
    effect_allele   : 'A1',
    strip_chr_prefix: false,
    train_only_freq : false,
    train_keep      : null,
    gene_bed_dir    : null,
    ggi_graph       : null,
]

// Keys with no sensible default -- the run cannot proceed without them.
def PRSNET_REQUIRED = ['gene_bed_dir', 'ggi_graph', 'base_gwas', 'ld_ref']

// params.outdir is not guaranteed to exist when launching with -params-file.
// publishDir with a null path warns and silently publishes nowhere.
def prsnetOutdir() {
    if (params.containsKey('outdir') && params.outdir)         return params.outdir
    if (params.containsKey('base_dir') && params.base_dir)     return "${params.base_dir}/out"
    return "${launchDir}/out"
}

def prsnetParam(String key) {
    def user = (params.prsnet instanceof Map) ? params.prsnet : [:]
    def value = user.containsKey(key) ? user[key] : PRSNET_DEFAULTS[key]
    if (value == null && key in PRSNET_REQUIRED) {
        error """
        Missing required parameter: params.prsnet.${key}

        The prsnet block belongs in workflows/config/dl_config.yaml, since that is
        what -params-file reads. Groovy-style `params.prsnet = [...]` only works in
        nextflow.config. Use literal paths -- \${...} does not interpolate in YAML:

          prsnet:
            base_gwas: /abs/path/data/supplement_data/sum_stats/alz.QC
            ld_ref: /abs/path/data/raw/UKB_ALZ/UKB_ALZ
            gene_bed_dir: /abs/path/data/supplement_data/prsnet/gene_bed_files_10kb
            ggi_graph: /abs/path/data/supplement_data/prsnet/ggi_graph.bin
        """.stripIndent()
    }
    return value
}



process PRSNET_BASE_QC {
    /*
     * Wraps bin/prsnet/run_base_qc.sh so the same implementation is used inside
     * Nextflow and for standalone pilot runs. The script prints a variant count
     * after every filter stage.
     */
    label 'process_medium'
    publishDir "${prsnetOutdir()}/prsnet/base_qc", mode: 'copy'

    input:
    path base_gwas
    path pheno
    tuple path(target_bed, stageAs: 'target/*'),
          path(target_bim, stageAs: 'target/*'),
          path(target_fam, stageAs: 'target/*')
    tuple path(ld_bed, stageAs: 'ldref/*'),
          path(ld_bim, stageAs: 'ldref/*'),
          path(ld_fam, stageAs: 'ldref/*')

    output:
    path "base_qc/gwas.QC.txt",                       emit: gwas_qc
    path "base_qc/SNP.pvalue",                        emit: snp_pvalue
    tuple path("base_qc/target_data.PH.bed"),
          path("base_qc/target_data.PH.bim"),
          path("base_qc/target_data.PH.fam"),         emit: target
    path "base_qc/snplist",                           emit: snplist
    path "base_qc/*.log",                             emit: logs

    script:
    def strip = prsnetParam('strip_chr_prefix') ? "--strip_chr_prefix" : ""
    """
    bash ${params.base_dir}/bin/prsnet/run_base_qc.sh \\
        --base_gwas ${base_gwas} \\
        --target target/${target_bed.baseName} \\
        --ld_ref ldref/${ld_bed.baseName} \\
        --pheno ${pheno} \\
        --outdir base_qc \\
        --min_maf ${prsnetParam('min_maf')} \\
        --min_info ${prsnetParam('min_info')} \\
        --target_maf ${prsnetParam('target_maf')} \\
        --target_hwe ${prsnetParam('target_hwe')} \\
        --target_geno ${prsnetParam('target_geno')} \\
        --target_mind ${prsnetParam('target_mind')} \\
        --memory ${prsnetParam('plink_memory')} \\
        --plink ${prsnetParam('plink_bin')} \\
        --plink2 ${prsnetParam('plink2_bin')} \\
        --effect_allele ${prsnetParam('effect_allele')} \\
        ${strip}
    """
}


process PRSNET_TRAIN_FREQ {
    /*
     * Allele frequencies from TRAINING samples only, fed to plink2 --read-freq so
     * that missing-genotype mean imputation inside --score never sees held-out
     * samples. Skip this and every gene PRS is mildly transductive.
     */
    label 'process_low'
    publishDir "${prsnetOutdir()}/prsnet/freq", mode: 'copy'

    input:
    tuple path(bed), path(bim), path(fam)
    path train_keep

    output:
    path "train_only.afreq", emit: freq

    when:
    prsnetParam('train_only_freq')

    script:
    """
    set -euo pipefail
    ${prsnetParam('plink2_bin')} --bfile ${bed.baseName} --keep ${train_keep} --freq \\
           --out train_only --memory ${prsnetParam('plink_memory')}
    """
}


process PRSNET_GENE_SNP_MAP {
    label 'process_medium'
    publishDir "${prsnetOutdir()}/prsnet/gene_map", mode: 'copy'

    input:
    path gwas_qc

    output:
    path "gene_order.txt",           emit: gene_order
    path "chunks/chunk_*.tsv",       emit: chunks
    path "gene_snp_map_stats.json",  emit: stats

    script:
    """
    python ${params.base_dir}/bin/prsnet/build_gene_snp_map.py \\
        --gwas_qc ${gwas_qc} \\
        --gene_bed_dir ${prsnetParam('gene_bed_dir')} \\
        --genes_per_chunk ${prsnetParam('genes_per_chunk')} \\
        --outdir .
    """
}


process PRSNET_SCORE_CHUNK {
    tag "chunk_${chunk.baseName}"
    label 'process_medium'

    input:
    tuple path(chunk), path(bed), path(bim), path(fam), path(gwas_qc), path(snp_pvalue), path(freq)

    output:
    path "${chunk.baseName}.npz", emit: scores
    path "${chunk.baseName}.log", emit: log

    script:
    def freq_arg = freq.name != 'NO_FREQ' ? "--freq_file ${freq}" : ""
    """
    python ${params.base_dir}/bin/prsnet/score_gene_chunk.py \\
        --chunk ${chunk} \\
        --bfile ${bed.baseName} \\
        --gwas_qc ${gwas_qc} \\
        --snp_pvalue ${snp_pvalue} \\
        --clump_r2 ${prsnetParam('clump_r2')} \\
        --clump_kb ${prsnetParam('clump_kb')} \\
        --plink_bin ${prsnetParam('plink_bin')} \\
        --plink2_bin ${prsnetParam('plink2_bin')} \\
        --threads ${task.cpus} \\
        --memory ${prsnetParam('plink_memory')} \\
        ${freq_arg} \\
        --output ${chunk.baseName}.npz \\
        --log ${chunk.baseName}.log
    """
}


process PRSNET_ASSEMBLE {
    label 'process_high_memory'
    publishDir "${prsnetOutdir()}/prsnet/features", mode: 'copy'

    input:
    path chunk_npz
    path gene_order
    tuple path(bed), path(bim), path(fam)
    path phenotype_file

    output:
    path "genotype_data.h5", emit: genotype_data
    path "phenotypes.csv",   emit: phenotypes
    path "data_stats.json",  emit: stats

    script:
    def chunk_list = chunk_npz.collect { it.name }.join(' ')
    def pheno_arg  = phenotype_file.name != 'NO_PHENO' ? "--phenotype_file ${phenotype_file}" : ""
    """
    python ${params.base_dir}/bin/prsnet/assemble_prsnet_features.py \\
        --chunks ${chunk_list} \\
        --gene_order ${gene_order} \\
        --fam ${fam} \\
        ${pheno_arg} \\
        --output_h5 genotype_data.h5 \\
        --output_pheno phenotypes.csv \\
        --stats_file data_stats.json
    """
}


process PRSNET_CONVERT_GRAPH {
    label 'process_low'
    publishDir "${prsnetOutdir()}/prsnet/graph", mode: 'copy'

    input:
    path gene_order

    output:
    path "ggi_graph.npz", emit: graph

    script:
    """
    python ${params.base_dir}/bin/prsnet/convert_ggi_graph.py \\
        --ggi_graph ${prsnetParam('ggi_graph')} \\
        --gene_bed_dir ${prsnetParam('gene_bed_dir')} \\
        --gene_order ${gene_order} \\
        --gene_order_out gene_order_from_graph.txt \\
        --allow_mismatch \\
        --truncate \\
        --output ggi_graph.npz

    # Fail loudly if the graph node order and the feature gene order disagree.
    if ! diff <(cut -f1 gene_order_from_graph.txt) ${gene_order} > /dev/null; then
        echo "FATAL: GGI node order != feature gene order. Genes would be wired to the wrong neighbours." >&2
        exit 1
    fi
    """
}


workflow PRSNET_FEATURES {
    take:
    base_gwas       // path
    target_plink    // tuple(bed, bim, fam)
    ld_plink        // tuple(bed, bim, fam)
    phenotype_file  // path

    main:
    PRSNET_BASE_QC(base_gwas, phenotype_file, target_plink, ld_plink)
    PRSNET_GENE_SNP_MAP(PRSNET_BASE_QC.out.gwas_qc)

    // PRSNET_SCORE_CHUNK always receives a freq slot, so an absent one needs a
    // real (empty) file to stage. Created on demand instead of relying on a
    // committed placeholder.
    if (prsnetParam('train_only_freq')) {
        def keep = prsnetParam('train_keep')
        if (!keep) {
            error "params.prsnet.train_only_freq is true but train_keep is not set. " +
                  "Point it at a file of training-sample IDs, or set train_only_freq: false."
        }
        freq_ch = PRSNET_TRAIN_FREQ(PRSNET_BASE_QC.out.target, file(keep, checkIfExists: true)).freq
    } else {
        def no_freq = file("${workDir}/prsnet_assets/NO_FREQ")
        if (!no_freq.exists()) {
            no_freq.parent.mkdirs()
            no_freq.text = ''
        }
        freq_ch = Channel.value(no_freq)
    }

    score_in = PRSNET_GENE_SNP_MAP.out.chunks
        .flatten()
        .combine(PRSNET_BASE_QC.out.target)
        .combine(PRSNET_BASE_QC.out.gwas_qc)
        .combine(PRSNET_BASE_QC.out.snp_pvalue)
        .combine(freq_ch)

    PRSNET_SCORE_CHUNK(score_in)

    PRSNET_ASSEMBLE(
        PRSNET_SCORE_CHUNK.out.scores.collect(),
        PRSNET_GENE_SNP_MAP.out.gene_order,
        PRSNET_BASE_QC.out.target,
        phenotype_file
    )

    PRSNET_CONVERT_GRAPH(PRSNET_GENE_SNP_MAP.out.gene_order)

    emit:
    genotype_data = PRSNET_ASSEMBLE.out.genotype_data
    phenotypes    = PRSNET_ASSEMBLE.out.phenotypes
    stats         = PRSNET_ASSEMBLE.out.stats
    gene_order    = PRSNET_GENE_SNP_MAP.out.gene_order
    graph         = PRSNET_CONVERT_GRAPH.out.graph
}
