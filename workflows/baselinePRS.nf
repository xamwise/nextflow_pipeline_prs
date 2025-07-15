include { quality_control } from '../modules/local/quality_control'
include { prsice } from '../modules/local/prsice'
include { sbayesr } from '../modules/local/sbayesr'
include { prs_csx } from '../modules/local/prs_csx'
include { prs_cs } from '../modules/local/prs_cs'
include { bridge_prs } from '../modules/local/bridge_prs'
include { prset } from '../modules/local/prset'
include { ldpred_funct } from '../modules/local/ldpred_funct'

include { baseline } from '../modules/local/baseline' 
include { sct } from '../modules/local/sct'
include { ldpred2 } from '../modules/local/ldpred2'
include ( lassosum ) from '../modules/local/lassosum'
// include { lassosum2 } from '../modules/local/lassosum2'
include { gaudi } from '../modules/local/gaudi'



workflow {
    quality_control_sum()
    quality_control()
    prsice()
    sbayesr()
    prs_csx()
    prs_cs()
    bridge_prs()
    prset()
    ldpred_funct()
    sct()
    ldpred2()
    lassosum()
    gaudi()
    baseline()
}