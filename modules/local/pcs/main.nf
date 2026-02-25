process pcs{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/pcs", mode: 'copy'

    input:
    val input
    val out
    val extract
    val pca

    output:
    val "${out}.eigenvec"

    script:
    """
    # Pruning before PCA
    plink --bfile $input --indep-pairwise 200 50 0.25 --out $input
    plink --bfile $input --extract $extract --pca $pca --out $out --memory 64000 --threads 8
    """
}