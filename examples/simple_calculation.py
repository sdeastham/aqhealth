from aqhealth import mortality
import numpy as np
import os
from datetime import datetime, timedelta
from gcgridobj import cstools

# gpw_dir:      directory must contain the GPW 4.11 population grids
# who_dir:      directory must contain the GHE2016 CSVs
# mort_preproc: directory must contain GPW_to_WHO.pkl and (if GEMM used) gemm_crf.csv
gpw_dir  = '/rds/general/user/seastham/home/GPWv4/GPWv4pt11'
who_dir  = '/rds/general/user/seastham/home/health_data/WHO_Data'
mort_preproc = '/rds/general/user/seastham/home/health_data/mort_preproc'
gemm_csv = os.path.join(mort_preproc,'gemm_crf.csv')
who_by_country = os.path.join(who_dir,'GHE2016_Deaths_2016-country.xls')
who_by_region  = os.path.join(who_dir,'GHE2016_Deaths_WHOReg_2000_2016.xls')

# Replace this with concentration data!
cs_res = 24
rng_seed = 19950407
rng = np.random.default_rng(seed=rng_seed)
pm_reference = np.zeros((6,cs_res,cs_res)) + rng.random((6,cs_res,cs_res)) * 10.0
pm_baseline = pm_reference + rng.random((6,cs_res,cs_res)) * 1.0
pm_policy = pm_baseline + rng.random((6,cs_res,cs_res)) * 5.0
cs_grid = cstools.gen_grid(cs_res)

# Each entry in chi_data corresponds to one simulation. For N simulations, we get N-1 results (difference relative to the reference simulation)
# Each simulation needs to have a grid specification
# Each simulation should have entries for every exposure factor, e.g.
# PM25: Annual mean PM2.5 (24h average), ug/m3
# O3_MDA8_ANN: Annuan mean MDA8 ozone (maximum daily average of 8-hour ozone, averaged over the year), ppbv
# O3_MDA1_SSN: Ozone-season mean (usually the 6 months during which that cell has the highest mean ozone) of the daily one-hour max ozone, ppbv
# Names for the exposure factors are dictated by the exposure_factor property of the ERFs in the mortality.py ERF class
# Here we have just two simulations (the minimum) and one exposure factor. We are also providing a 
# "reference" s
chi_data = {'reference': {'PM25': pm_reference, 'grid': cs_grid},
            'baseline':  {'PM25': pm_baseline, 'grid': cs_grid},
            'policy':    {'PM25': pm_policy, 'grid': cs_grid}}

# Which concentration set is the reference one?
ref_map = 'reference'

# What exposure response functions do we want to use?
# Either define by hand or...
ERF_set = [mortality.erf_lib['chen_hoek_2020_AC']]

# ...use the GEMM set (you can append more, and indeed will need to to add ozone ERFs)
ERF_set_full = mortality.gen_GEMM_erfs(gemm_csv)
# Only keep the NCD+LRI ERFs
ERF_set = [x for x in ERF_set_full if '_NCD+LRI_' in x.name]

# Number of samples (usually need at least 1000 - using 10 for demo, but
# for a real case you should go higher and check convergence)
n_samples = 10

result_data = mortality.calc_mortality(chi_data, ref_map, common_chi_grid=True, ERF_set=ERF_set, GPW_dir=gpw_dir, WHO_dir=who_dir, n_rand=n_samples, mort_preprocessor_dir=mort_preproc)

# Check the results against the expected value from a previous calculation
total_mort = 0.0
for key in result_data['PM25'].keys():
    age_range = key.split('_')[-1]
    age_morts_mean = np.mean(np.sum(result_data['PM25'][key]['policy']['morts_vs_ref'],axis=0))
    total_mort += age_morts_mean
    print(f'{age_range:10s}: {age_morts_mean:10.1f}')
print('')
tot_str = 'TOTAL'
print(f'{tot_str:10s}: {total_mort:10.1f}')

# Evaluate against pre-calculated value
precalc_mort = 1397759.1
if np.abs(total_mort - precalc_mort) > 1.0:
    print(f'Warning: result differs from expected value of {expected_val:10.1f}')
else:
    print(f'Test passed.')
