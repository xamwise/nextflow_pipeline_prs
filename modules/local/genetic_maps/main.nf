/*
 * Downloads the 1000 Genomes OMNI genetic maps used by bigsnpr::snp_asGeneticPos() (LassoSum2, LDpred2)
 * once into a shared folder. Without it every task downloads all 22 maps (~300 MB) into its own work
 * directory, and parallel tasks (e.g. one per fold) make GitHub time out.
 * Maps that are already in the folder are not downloaded again.
 */
process genetic_maps {

    label 'process_single'

    input:
    val out_dir

    output:
    val out_dir

    script:
    """
    set -euo pipefail
    mkdir -p ${out_dir}

    for chr in \$(seq 1 22); do
        map=${out_dir}/chr\${chr}.OMNI.interpolated_genetic_map
        if [ ! -s \$map ]; then
            curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors \\
                https://github.com/joepickrell/1000-genomes-genetic-maps/raw/master/interpolated_OMNI/chr\${chr}.OMNI.interpolated_genetic_map.gz \\
                | gunzip > \$map.tmp
            mv \$map.tmp \$map
        fi
    done
    """
}
