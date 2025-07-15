process prset {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/prset", mode: 'copy'

    input:
    val base
    val pheno
    val target
    val out
    val snp
    val a1
    val a2
    val stat
    val beta
    val pvalue
    val binary_target
    val bar_levels
    val no_full
    val fastscore
    val bed


    output:
    out

    script:
    """
    

    Rscript /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/PRSice.R \\
        --prsice PRSice_mac  \\
        --base $base \\
        --target $target \\
        --A1 $a1 \\
        --A2 $a2 \\
        --snp MarkerName \\
        --pvalue $pvalue \\
        --stat $stat \\
        --beta $beta \\
        --binary-target $binary_target \\
        --pheno Target_Data/TAR.height \\
        --cov Target_Data/TAR.covariate \\
        --out Height.set \\
        --gtf $gtf \\
        --wind-5 5kb \\
        --wind-3 1kb \\
        --msigdb $msigdb \\
        --multi-plot $quantiles \\
        --set-perm 1000 \\
        --out $out 
    """

}