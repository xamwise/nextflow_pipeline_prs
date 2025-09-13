process baseline {

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/baseline", mode: 'copy'

    input:
    path file, sup_data

    output:
    path('baseline*.html'), emit: baseline, optional: true

    script:
    """
    Rscript ${params.base_dir}/bin/PRS.R \\ 
        --file $file \\ 
        --sup_data $sup_data
    """

}