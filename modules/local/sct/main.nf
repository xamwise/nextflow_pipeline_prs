process sct {
    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sct", mode: 'copy'

    input:
    path ref_dir, path sst_file
    
    output:
    path('sct*.html'), emit: sct, optional: true

    script:
    """
    Rscript SCT.R \\
    --file ref_dir \\
    --sumstats sst_file
     """
}