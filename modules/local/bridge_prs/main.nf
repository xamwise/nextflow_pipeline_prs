process bridge_prs {
    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/bridge_prs", mode: 'copy'

    input:
    
    output:
    path('bridge_prs*.html'), emit: prs_cs, optional: true

    script:
    """
    bridgePRS prs-single run \\
        -o out_single/ \\
        --config_file data/eur_eas.config \\
        --phenotype y \\
        --cores 4 
    """
}