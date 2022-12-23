import multiprocessing
import numpy as np
import os
import pandas as pd
import mcstats
import pandas
from typing import List

# WARNING: with slurm, it seems that you want
# to use CPUS PER TASK and not TASKS PER NODE
# So the batch script should use -n 1 -c 8

# Needed for a full mortality calculation
import netCDF4
import pickle
try:
    from tqdm import tqdm
    use_tqdm = True
except:
    use_tqdm = False
from gcgridobj import latlontools, regrid

# This is VERSION 3 of the mortality code
# It includes multiprocessor support, and
# is derived from mortality_mp_v2

# Use a global array to deal with shared memory
var_dict={}
def init_worker(i_batch,n_per_batch,n_draws,n_country,ERF,pop_inflator,unique_cids,
                chi_scen_R, chi_ref_R, chi_ref_raw_R, baseline_incidence_R, pop_grid_R,
                chi_delta_raw_R, cid_grid_R, regionid_grid_R, regionid_list, n_region):
    # Need the shape of the arrays for later manipulation
    #var_dict['world_shape'] = world_shape

    # Simple data - not in special arrays
    var_dict['i_batch']      = i_batch
    var_dict['n_per_batch']  = n_per_batch
    var_dict['n_draws']      = n_draws
    var_dict['n_country']    = n_country
    var_dict['n_region']     = n_region
    var_dict['ERF']          = ERF
    var_dict['pop_inflator'] = pop_inflator
    var_dict['unique_cids']  = unique_cids

    # Big data - all raw (R) arrays
    var_dict['chi_scen']           = chi_scen_R
    var_dict['chi_ref']            = chi_ref_R
    var_dict['chi_ref_raw']        = chi_ref_raw_R
    var_dict['baseline_incidence'] = baseline_incidence_R
    var_dict['pop_grid']           = pop_grid_R
    var_dict['chi_delta_raw']      = chi_delta_raw_R
    var_dict['cid_grid']           = cid_grid_R
    var_dict['region_masks']       = regionid_grid_R
    var_dict['region_ids']         = regionid_list

def worker_calc(i_in_batch):
    # This will be run on a worker which has access to var_dict; see
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

    # Simple data
    i_batch            = var_dict['i_batch']
    n_per_batch        = var_dict['n_per_batch']
    n_draws            = var_dict['n_draws']
    n_country          = var_dict['n_country']
    n_region           = var_dict['n_region']
    ERF                = var_dict['ERF']
    pop_inflator       = var_dict['pop_inflator']
    unique_cids        = var_dict['unique_cids']
    region_ids         = var_dict['region_ids']   # List: for each CID, the region IDs associated with it (list of lists)

    # Big data
    chi_scen           = np.frombuffer(var_dict['chi_scen'])
    chi_ref            = np.frombuffer(var_dict['chi_ref'])
    chi_ref_raw        = np.frombuffer(var_dict['chi_ref_raw'])
    baseline_incidence = np.frombuffer(var_dict['baseline_incidence'])
    pop_grid_dbl       = np.frombuffer(var_dict['pop_grid'])
    chi_delta_raw      = np.frombuffer(var_dict['chi_delta_raw'])
    cid_grid           = np.frombuffer(var_dict['cid_grid']).astype(np.int)
    # Masks of subregions (dimensions of cid_grid)
    get_regions = n_region > 0
    if get_regions:
        region_masks   = np.frombuffer(var_dict['region_masks']).astype(np.int)
    

    i_draw = i_in_batch + (i_batch*n_per_batch)
    if i_draw >= n_draws:
        return 0.0, None, None, None

    get_extra = i_in_batch == 0

    if get_extra:
        extra_data = {'delta_exp':     np.zeros(n_country),
                      'base_exp':      np.zeros(n_country),
                      'base_morts':    np.zeros(n_country),
                      'affected_pop':  np.zeros(n_country)}
    else:
        extra_data = None

    # Calculate the relative risk for every grid cell
    rr_scen = ERF.erf.calc_rr(chi_scen,i_sample=i_draw,run_all=False)[0]
    rr_obs  = ERF.erf.calc_rr(chi_ref,i_sample=i_draw,run_all=False)[0]
    rr_ratio = rr_scen/rr_obs

    # Need to aggregate observed incidence * (rr_ratio - 1). This is an 
    # estimate of the number of mortalities relative to the observed 
    # ("reference") scenario
    draw_morts = np.zeros(n_country)
    region_morts = np.zeros(n_region)
    for i_country in range(n_country):
        #cid_mask = cid_precalc_int[i_country,...] > 0
        cid_mask = cid_grid == unique_cids[i_country]
        draw_morts[i_country] = np.nansum(pop_inflator[i_country] * baseline_incidence[cid_mask] * (rr_ratio[cid_mask] - 1.0))
        # Are we running regional calculations?
        if region_ids is not None:
            n_subregion = len(region_ids[i_country])
            if n_subregion > 0:
                for region_id in region_ids[i_country]:
                    region_mask = region_masks == region_id
                    region_morts[region_id] = np.nansum(pop_inflator[i_country] * baseline_incidence[region_mask] * (rr_ratio[region_mask] - 1.0))
        if i_in_batch == 0:
            extra_data['delta_exp'][i_country]    = np.nansum(pop_inflator[i_country] * pop_grid_dbl[cid_mask] * chi_delta_raw[cid_mask])
            extra_data['base_exp'][i_country]     = np.nansum(pop_inflator[i_country] * pop_grid_dbl[cid_mask] * chi_ref_raw[cid_mask])
            extra_data['base_morts'][i_country]   = np.nansum(pop_inflator[i_country] * baseline_incidence[cid_mask])
            extra_data['affected_pop'][i_country] = np.nansum(pop_inflator[i_country] * pop_grid_dbl[cid_mask])
    return draw_morts, region_morts, extra_data, i_in_batch

# This module is designed to handle mortality calculations
# It defines common exposure response functions

# Exposure response functions
class base_erf:
    @property
    def form(self):
        return 'base'
    
    def __str__(self):
        return('ERF ({}) covering {:d} random draws'.format(self.form,self.n_rand))
    
    def __repr__(self):
        return('base_erf(n_rand={:d})'.format(self.n_rand))
    
    def __init__(self,*args,**kwargs):
        # Set up the properties. Note that the
        # update_params method is overwritten in 
        # basically all subclasses
        self.update_params(*args,**kwargs)
    
    def update_params(self,n_rand : int):
        self.n_rand = n_rand
        self.erf = lambda chi, i : np.zeros(chi.shape) + 1.0
    
    def process_chi(self,chi):
        # Transform the forcing based on the form
        # of the exposure response function. IMPORTANT:
        # assumes that the transform parameters do not
        # vary between random draws!
        return chi
    
    def calc_rr(self,chi_tfm,i_sample : int = 0, run_all : bool = True) -> List:
        # Must return the risk of chi_tfm relative to some counterfactual value
        if run_all:
            # Ignore i_sample and run all possible beta values
            return [self.calc_rr(chi_tfm,i,run_all=False)[0] for i in range(self.n_rand)]
        else:
            # Run a single beta value
            return [self.erf(chi_tfm,i_sample)]

class log_linear_erf(base_erf):
    @property
    def form(self):
        return 'log-linear'
    
    def __repr__(self):
        return('log_linear_erf(beta={},chi_min={})'.format(self.beta,self.chi_min))
    
    def update_params(self,beta,chi_min=None):
        self.beta = np.asanyarray(beta)
        self.n_rand = self.beta.size
        if chi_min is not None:
            self.chi_min = chi_min
        self.erf = lambda chi_tfm, i : np.exp(self.beta[i]*chi_tfm)
        
    def process_chi(self,chi):
        return np.maximum(np.asanyarray(chi) - self.chi_min,0.0)

class power_erf(base_erf):
    @property
    def form(self):
        return 'power'
    
    def __repr__(self):
        return('power_erf(c={},chi_min={})'.format(self.c,self.chi_min))

    def update_params(self,beta,chi_min=None):
        # "beta" here is c, usually 0.6 (bounds: 0.2-1.0)
        # Mortality ~ dose ^ c
        self.beta = np.asanyarray(beta)
        self.n_rand = self.beta.size
        if chi_min is not None:
            self.chi_min = chi_min
        self.erf = lambda chi_tfm, i : np.power(chi_tfm,self.beta[i])
    
    def process_chi(self,chi):
        # "Minimum dose" would be problematic here
        return np.maximum(np.asanyarray(chi) - self.chi_min,0.0)

# Should use (% change per 1 ug/m3) = 100*((1/chi)*A + B)
class vodonos_erf(base_erf):
    @property
    def form(self):
        return 'vodonos'
    
    def __repr__(self):
        return('vodonos_erf(A={},B={},chi_min={})'.format(self.A,self.B,self.chi_min))
    
    def update_params(self,A,B,chi_min=None):
        self.A = np.asanyarray(A)
        self.B = np.asanyarray(B)
        self.n_rand = self.A.size
        #print('VODONOS SIZE',self.A.size,self.B.size)
        if chi_min is None:
            chi_min = 0.0
        self.chi_min = chi_min
        #assert self.B.size == self.n_rand, 'Both parameters must have the same number of parameters'
        self.erf = lambda chi_tfm, i : self.vodonos_calc(chi_tfm,i)

    def vodonos_calc(self,chi,i):
        # Inverse transformation of PM2.5 with intercept gives fractional increase in mortality per 1 ug/m3 increase in PM2.5
        # Get a relative risk as 1.0 + (fractional increase)
        beta = np.log(1.0 + (self.A[i]/chi) + self.B[i])/1.0
        return np.exp(beta * chi)
        
    def process_chi(self,chi):
        return np.maximum(np.asanyarray(chi) - self.chi_min,0.0)
    
class GEMM_erf(log_linear_erf):
    # This should be the GEMM for a single cause of 
    # death and for a single age range
    @property
    def form(self):
        return 'GEMM'
    
    def __repr__(self):
        return('GEMM_erf(theta={},alpha={},mu={},nu={},chi_min={})'.format(self.theta,self.alpha,self.mu,self.nu,self.chi_min))
    
    def update_params(self,theta,alpha=None,
                 mu=None, nu=None,
                 chi_min=None):
        if alpha is not None:
            self.alpha = alpha
        if mu is not None:
            self.mu = mu
        if nu is not None:
            self.nu = nu
        super().update_params(beta=theta,chi_min=chi_min)
    
    def process_chi(self,chi):
        return calc_GEMM_transform(chi,self.chi_min,self.alpha,self.mu,self.nu)

class log_log_erf(base_erf):
    @property
    def form(self):
        return 'log-log'
    
    def __repr__(self):
        return('log_log_erf(beta={},chi_min={})'.format(self.beta,self.chi_min))
    
    def update_params(self,beta,chi_min=None):
        self.beta = np.asanyarray(beta)
        if chi_min is not None:
            self.chi_min = chi_min
        self.n_rand = self.beta.size
        self.erf = lambda chi_tfm, i : np.power((chi_tfm+1)/(self.chi_min+1),self.beta[i])
    
class IER_erf(base_erf):
    @property
    def form(self):
        return 'IER'
    
    def __repr__(self):
        return('IER_erf()')
    
    def __init__(self,*args,**kwargs):
        raise ValueError('IER not implemented')

# A class to hold an ERF
class meta_erf:
    def __init__(self,erf,disease_name,disease_data,exposure_factor,
                 param_gen,min_age=0,max_age=np.inf,name=None):
        # A base_erf class object. For GEMM, this might be:
        # erf = mortality.GEMM_erf(theta=np.nan,
        #                          alpha=row['alpha'],
        #                          mu=row['mu'],
        #                          nu=row['nu'],
        #                          chi_min=2.4)
        # In this case, we are assuming that alpha, mu, and nu
        # are read from a pandas dataframe. Theta is the parameter
        # which will be set using param_gen
        self.erf = erf
        # Function which map from a uniform random variable over
        # the interval 0 to 1 to the distribution used by the ERF.
        # This MUST be a list of generators, even if only length 1.
        # For GEMM, this might be:
        # param_gen = [mcstats.norm_var(mean=theta_mean,sdev=theta_std)]
        if not isinstance(param_gen,list):
            self.param_gen = [param_gen]
        else:
            self.param_gen = param_gen
        self.n_rand_params = len(self.param_gen)
        # Integers; inclusive (e.g. 25 - 39 means 25 <= age < 40)
        self.min_age = min_age
        self.max_age = max_age
        # This is the disease's name (for user convenience only)
        self.disease_name = disease_name
        # This should be a "standard_disease" class object
        self.disease_data = disease_data
        # The exposure causing the issue - e.g. PM25, O3, or UV
        self.exposure_factor = exposure_factor
        # Need an identifier
        if name is None:
            name = '{:s}_{:s}_{:.0f}to{:.0f}'.format(exposure_factor,disease_name,min_age,max_age)
        self.name = name
        # Need an internal counter
        self.i_param = (self.n_rand_params - 1)
    
    def __repr__(self):
        return('meta_erf(erf={},disease_name={},disease_data={},exposure_factor={},param_gen={},min_age={},max_age={})'.format(
            self.erf,self.disease_name,self.disease_data,self.exposure_factor,self.param_gen,self.min_age,self.max_age))
    
    def __str__(self):
        return('Implementation of {} for age range {} -> {}, disease {:s}, factor {:s}. ID: {:s}'.format(
            self.erf,self.min_age,self.max_age,self.disease_name,self.exposure_factor,self.name))
    
    def generate(self,uni_vec,i_param=None,renew=False):
        # Accepts an array of uniformly-distributed random
        # samples from 0 to 1
        if i_param is None:
            self.i_param = (self.i_param + 1)%self.n_rand_params
            i_param = self.i_param
        self.param_gen[i_param].generate(uni_vec)
        if renew:
            self.update_erf_params()

    def update_erf_params(self):
        self.erf.update_params(*[self.param_gen[x_param].provide(None) for x_param in range(self.n_rand_params)])
        
# Miscellaneous functions
def calc_GEMM_transform(chi,chi_min,alpha,mu,nu):
    ''' GEMM transform of concentrations
    Approach below prevents calculation being performed
    on elements which are below the threshold'''
    chia = np.asanyarray(chi)
    chi_greater = chia > chi_min
    tfm = np.zeros(chia.shape)
    z = chia[chi_greater] - chi_min
    tfm[chi_greater] = np.log((z/alpha) + 1.0)/(1.0 + np.exp(-1.0*(z - mu)/nu))
    
    # Shorter but potentially slower equivalent:
    #z = np.maximum(np.asanyarray(chi) - chi_min, 0.0)
    #return np.where(z>0,np.log((z/alpha) + 1.0)/(1.0 + np.exp(-1.0*(z - mu)/nu)),1.0)
    
    return tfm

def read_GEMM(csv_path):
    ''' Read in the GEMM data, as available at https://github.com/mszyszkowicz/DataGEMM.git'''
    # Currently actually using the data as pulled direct from the GEMM paper
    # Code developed (and data pulled) by Guillaume Chossiere
    GEMM_data = pd.read_csv(csv_path)
    
    # Remove the average values (correspond to the entries "age >25") when
    # there is age-specific data, otherwise just keep the ">25" value
    to_drop = []
    for cause in set(GEMM_data.cause):
        if len(GEMM_data.loc[GEMM_data.cause == cause]) > 1:
            to_drop.append(GEMM_data.query(f'cause == "{cause}" and age == ">25"').index.values[0])
    GEMM_data.drop(to_drop, axis=0, inplace=True)
    
    return GEMM_data

def gen_GEMM_erfs(csv_path='GEMM/gemm_crf.csv',use_norm=False):
    erf_set = []
    GEMM_data = read_GEMM(csv_path)
    for idx, row in GEMM_data.iterrows():
        cause = row['cause']
        age = row['age']
        if age == '>25':
            min_age = 25
            max_age = np.inf
        elif age == '80+':
            min_age = 80
            max_age = np.inf
        else:
            age_split = age.split('-')
            min_age = int(age_split[0])
            max_age = int(age_split[1])
        # Mean and standard error
        theta_mean = row['theta']
        theta_std  = row['std theta']
        if use_norm:
            param_gen = [mcstats.norm_var(mean=theta_mean,
                                          sdev=theta_std)]
        else:
            # Fit a triangular distribution to give the 
            # same 95% CI
            theta_025 = theta_mean - (1.96*theta_std)
            theta_975 = theta_mean + (1.96*theta_std)
            theta_low,theta_high,h = mcstats.iter_tri_bounds(theta_025,
                                                             theta_mean,
                                                             theta_975,
                                                             95)
            param_gen = [mcstats.tri_var(low=theta_low,
                                         mid=theta_mean,
                                         high=theta_high)]
        erf = GEMM_erf(theta=np.nan,
                       alpha=row['alpha'],
                       mu=row['mu'],
                       nu=row['nu'],
                       chi_min=2.4)
        # Look through the GEMM inventory
        assert cause in GEMM_diseases.keys(), 'Disease {:s} not in GEMM disease dictionary'.format(cause)
        
        erf_set.append(meta_erf(name='GEMM_{:s}_{:s}'.format(cause,age),
                                erf=erf,
                                disease_name=cause,
                                disease_data=GEMM_diseases[cause],
                                exposure_factor='PM25',
                                param_gen=param_gen,
                                min_age=min_age,
                                max_age=max_age))
                                        #param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])
    return erf_set

def calc_rr_ratio(aff_pop,erf_obj,outcome_obs,conc_base,conc_ptb,conc_obs=None):
    '''Calculate mortality impact of a change from a baseline to perturbed state'''
    # erf_obj must be of the erf class
    # Use the ERF's map_rand function to transform from a value between 0 and 1 to
    # the parameters needed by the ERF
    erf_params = erf_obj.map_rand(random_vars)
    
    # Delta I/observed I
    delta_inc_ratio = erf(beta=beta,chi_base=conc_base,chi_ptb=conc_ptb,chi_obs=conc_obs)
    
    return delta_mort

def read_who_data(who_by_country_xls=None,who_by_region_xls=None,as_rates=False):
    # Open the WHO "by country" XLS
    if who_by_country_xls is None:
        who_by_country_xls = os.path.join('/net/d13/data/seastham',
                                          'HealthData/WHO_Mortality',
                                          'WHO2016/GHE2016_Deaths_2016-country.xls')
    if who_by_region_xls is None:
        who_by_region_xls  = os.path.join('/net/d13/data/seastham',
                                          'HealthData/WHO_Mortality',
                                          'WHO2016/GHE2016_Deaths_WHOReg_2000_2016.xls')
    
    who_data = {}
    for age_code in ['0-4','5-14','15-29','30-49','50-59','60-69','70+']:
        # Get by-country first
        who_xls = pandas.read_excel(who_by_country_xls,
                                    sheet_name='Deaths ' + age_code,
                                    header=7,usecols=[0,1] + list(range(7,1000)))
        # Rename the columns
        who_xls = who_xls.rename(columns={'Unnamed: 0': 'Sex', 'Unnamed: 1': 'GHE Code'})
        # Keep only the unisex data
        who_xls = who_xls[who_xls['Sex'] == 'Persons']
        who_xls = who_xls.reset_index(drop=True)
        for col_name, col_data in who_xls.items():
            if col_name not in ['Sex','GHE Code']:
                temp_data = pandas.to_numeric(col_data,errors='coerce')
                # Leave NaNs unchanged
                #temp_data[np.isnan(temp_data)] = 0.0
                who_xls[col_name] = temp_data
        # This is actually "population"
        who_xls.at[0,'GHE Code'] = -1
        who_data[age_code] = who_xls.copy()

    # Now read in regional data
    for reg in ['Afr','Amr','Sear','Eur','Emr','Wpr','Global']:
        who_xls = pandas.read_excel(who_by_region_xls,
                            sheet_name='2016 ' + reg,
                            header=7,usecols=[0] + list(range(9,1000)))
        who_xls = who_xls.rename(columns={'Unnamed: 0': 'GHE Code'})
        who_xls = who_xls.drop(index=[1,2]).reset_index(drop=True)
        
        for age_range in ['0-28 days','1-59 months','5-14','15-29',
                          '30-49','50-59','60-69','70+']:
            if 'days' in age_range or 'months' in age_range:
                in_col = age_range
                out_col = '0-4'
            else:
                in_col = age_range + ' years'
                out_col = age_range
            if reg not in who_data[out_col].columns:
                who_data[out_col][reg] = 0.0
            for suffix in ['','.1']:
                temp_vec = who_xls[in_col + suffix].values
                # Make everything in terms of thousands
                temp_vec[1:] *= 0.001
                who_data[out_col][reg] += temp_vec
    
    # Last-minute processing
    # Change GHE codes to integers to avoid comparison SNAFUs
    # If requested, also convert all data from absolute mortality counts to mortalities per capita
    for age_range, data in who_data.items():
        for loc, col in data.items():
            if loc == 'GHE Code':
                temp_data = np.asarray(col).astype(np.int32)
                who_data[age_range][loc] = temp_data.copy()
            if as_rates and (loc not in ['Sex','GHE Code']):
                temp_data = np.asarray(col)
                temp_data[1:] /= temp_data[0]
                who_data[age_range][loc] = temp_data.copy()

    return who_data
        
class standard_disease():
    def __init__(self,ghe_codes=None):
        self.ghe_codes = ghe_codes
    def id_GHEs(self,who_data):
        # Return the indices needed for this ERF
        who_sheets = list(who_data.keys())
        # Use the first sheet just to get a list of GHE codes
        ghe_list = who_data[who_sheets[0]]['GHE Code']
        # Return a list of booleans which can be used to index
        return [x in self.ghe_codes for x in ghe_list]
    def calc_rates(self,who_data,sum_op=np.nansum):
        # Calculate the net mortality rate for each country/region
        who_sheets = list(who_data.keys())
        # Use the first sheet just to get a list of GHE codes
        ghe_list = who_data[who_sheets[0]]['GHE Code']
        targ_idx = [x in self.ghe_codes for x in ghe_list]
        rates = {}
        for sheet in who_sheets:
            rates[sheet] = {}
            for loc, col in who_data[sheet].items():
                if loc not in ['Sex','GHE Code']:
                    rates[sheet][loc] = sum_op(col[targ_idx])
        return rates

# Some typical ones...
standard_diseases = {'All-cause':                    standard_disease([0]),
                     'Non-communicable disease':     standard_disease([600]),
                     'Nonaccidental':                standard_disease([10,600]), # Excludes injuries
                     'Lung cancer':                  standard_disease([680]),
                     'Malignant melanoma':           standard_disease([691]),
                     'Cardiopulmonary disease':      standard_disease([1100,1170]),
                     'Cardiovascular disease':       standard_disease([1100]),
                     'Respiratory disease':          standard_disease([390,400,1170]),
                     'COPD + asthma':                standard_disease([1180,1190]),
                     'Lower respiratory infections': standard_disease([390]),
                     'Stroke':                       standard_disease([1140]),
                     'Ischaemic stroke':             standard_disease([1141]),
                     'Ischaemic heart disease':      standard_disease([1130]),
                     'COPD':                         standard_disease([1180]),
                     'Diabetes':                     standard_disease([800]),}
                     

# Definitions used by Turner et al (2015)
turner_diseases = {'All-cause':              standard_disease([0]),
                   'Circulatory + diabetes': standard_disease([1100,800]),
                   'Respiratory disease':    standard_disease([390,400,1170]),
                   'COPD plus':              standard_disease([390,1180,1190])}

# Definitions used for the GEMM
GEMM_diseases = {'Stroke':      standard_disease([1140]),
                 'IHD':         standard_disease([1130]),
                 'LRI':         standard_disease([390]),
                 'Lung Cancer': standard_disease([680]),
                 'COPD':        standard_disease([1180]),
                 'NCD':         standard_disease([600]),
                 'NCD+LRI':     standard_disease([600,390])}
    

def convert_rrs_to_beta_se(rr_low,rr_mid,rr_high,interval,increment):
    # Based on method reported from Chen and Hoek (2020)
    std_err   = (np.log(rr_high) - np.log(rr_low))/(2.0*1.96*increment)
    beta_mid  = np.log(rr_mid)/increment
    beta_low  = beta_mid - 1.96 * std_err
    beta_high = beta_mid + 1.96 * std_err
    return beta_low, beta_mid, beta_high

def convert_rrs_to_beta_tri(rr_low,rr_mid,rr_high,interval,increment):
    # Based on assuming a triangular distribution
    std_err   = (np.log(rr_high) - np.log(rr_low))/(2.0*1.96*increment)
    beta_mid  = np.log(rr_mid)/increment
    beta_low  = beta_mid - 1.96 * std_err
    beta_high = beta_mid + 1.96 * std_err
    return beta_low, beta_mid, beta_high


# Library of ERFs
erf_lib = {}
# Chen and Hoek 2020, all-cause
rr = (1.06,1.08,1.09)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['chen_hoek_2020_AC'] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='Non-accidental',
                                        disease_data=standard_diseases['Nonaccidental'],
                                        exposure_factor='PM25',
                                        name='Chen and Hoek non-accidental',
                                        min_age=0,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])
# Hoek et al 2013, cardiovascular
rr = (1.05,1.11,1.16)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['hoek_meta_2013_CV'] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='Cardiovascular',
                                        disease_data=standard_diseases['Cardiovascular disease'],
                                        exposure_factor='PM25',
                                        name='Hoek meta-analysis cardiovascular',
                                        min_age=30,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])
# EPA 2011, all-cause
rr = (1.004,1.01,1.018)
beta_025, beta_mean, beta_975 = np.log(rr)/1.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['epa_2011_AC'      ] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='All-cause',
                                        disease_data=standard_diseases['All-cause'],
                                        exposure_factor='PM25',
                                        name='EPA 2011 all-cause',
                                        min_age=30,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Krewski et al 2009 all-cause
# Krewski, D., Jerrett, M., Burnett, R. T., Ma, R., Hughes, E., Shi, Y., Turner, M. C., Pope, C. A. I., Thurston, G., Calle, E. E. and Thun, M. J.: Extended follow-up and spatial analysis of the American Cancer Society study linking particulate air pollution and mortality. HEI Research Report 140, Health Effects Institute, Boston, MA., 2009.
rr = (1.04,1.06,1.08)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['krewski_2009_AC'  ] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='All-cause',
                                        disease_data=standard_diseases['All-cause'],
                                        exposure_factor='PM25',
                                        name='Krewski ACS 2009 all-cause',
                                        min_age=30,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Jerrett et al 2009 respiratory
rr = (1.010,1.040,1.067)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['jerrett_2009_RD'  ] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='Respiratory disease',
                                        disease_data=standard_diseases['Respiratory disease'],
                                        exposure_factor='O3_MDA1_SSN',
                                        name='Jerrett ACS 2009 respiratory',
                                        min_age=30,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Jerrett et al 2009 respiratory (COPD + asthma ONLY)
rr = (1.010,1.040,1.067)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['jerrett_2009_NIRD'] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='Respiratory disease',
                                        disease_data=standard_diseases['COPD + asthma'],
                                        exposure_factor='O3_MDA1_SSN',
                                        name='Jerrett ACS 2009 COPD + asthma',
                                        min_age=30,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Turner et al 2015 respiratory
rr = (1.08,1.12,1.16)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['turner_2015_RD'   ] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='Respiratory disease',
                                        disease_data=turner_diseases['Respiratory disease'],
                                        exposure_factor='O3_MDA8_ANN',
                                        name='Turner 2015 respiratory',
                                        min_age=30,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Jerrett et al 2009 respiratory (threshold from study)
rr = (1.010,1.040,1.067)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['jerrett_2009_RD_thr'] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=56.0),
                                         disease_name='Respiratory disease',
                                         disease_data=standard_diseases['Respiratory disease'],
                                         exposure_factor='O3_MDA1_SSN',
                                         name='Jerrett ACS 2009 respiratory thresholded',
                                         min_age=30,
                                         max_age=np.inf,
                                         param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Jerrett et al 2009 respiratory (COPD + asthma ONLY) (threshold from study)
rr = (1.010,1.040,1.067)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['jerrett_2009_NIRD_thr']   = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=56.0),
                                              disease_name='Respiratory disease',
                                              disease_data=standard_diseases['COPD + asthma'],
                                              exposure_factor='O3_MDA1_SSN',
                                              name='Jerrett ACS 2009 COPD + asthma thresholded',
                                              min_age=30,
                                              max_age=np.inf,
                                              param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Turner et al 2015 respiratory (threshold from study - not recommended)
rr = (1.08,1.12,1.16)
beta_025, beta_mean, beta_975 = np.log(rr)/10.0
beta_low,beta_high,h = mcstats.iter_tri_bounds(beta_025,beta_mean,beta_975,95)
erf_lib['turner_2015_RD_thr'] = meta_erf(erf=log_linear_erf(beta=np.nan,chi_min=35.0),
                                         disease_name='Respiratory disease',
                                         disease_data=turner_diseases['Respiratory disease'],
                                         exposure_factor='O3_MDA8_ANN',
                                         name='Turner 2015 respiratory thresholded',
                                         min_age=30,
                                         max_age=np.inf,
                                         param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

# Vodonos et al 2018
# Should use (% change per 1 ug/m3) = 100*((1/chi)*A + B)
# For all cause, A is 0.071 (SE: 0.038) < Used for cause-specific, too
#                B is 0.006 (SE: 0.003)
A_mid  = 0.071
A_SE   = 0.038
A_low  = A_mid - 1.96 * A_SE
A_high = A_mid + 1.96 * A_SE

B_mid  = 0.006
B_SE   = 0.003
B_low  = B_mid - 1.96 * B_SE
B_high = B_mid + 1.96 * B_SE
erf_lib['vodonos_2018_AC'  ] = meta_erf(erf=vodonos_erf(A=np.nan,B=np.nan,chi_min=0.0),
                                        disease_name='All-cause',
                                        disease_data=standard_diseases['All-cause'],
                                        exposure_factor='PM25',
                                        name='Vodonos 2018 all-cause',
                                        min_age=14, # Was 0
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=A_low,mid=A_mid,high=A_high),
                                                   mcstats.tri_var(low=B_low,mid=B_mid,high=B_high)])


# Slaper et al 1996 skin cancer
# Use 0.6 +/- 0.4 as the full range
beta_low, beta_mean, beta_high = (0.2,0.6,1.0)
erf_lib['slaper_1996_SC'   ] = meta_erf(erf=power_erf(beta=np.nan,chi_min=0.0),
                                        disease_name='Malignant melanoma',
                                        disease_data=standard_diseases['Malignant melanoma'],
                                        exposure_factor='UV_SCUPh',
                                        name='Slaper 1996 skin cancer',
                                        min_age=0,
                                        max_age=np.inf,
                                        param_gen=[mcstats.tri_var(low=beta_low,mid=beta_mean,high=beta_high)])

def calc_mortality(chi_data, ref_map, common_chi_grid, ERF_set, GPW_dir, WHO_dir, n_rand, rng_opts={'rng_name': 'pseudorandom', 'random_seed': 20201112}, verbose=True, US_states=False, state_mask_file=None, mort_preprocessor_dir='',n_cpus=None):
    # How many scenarios are there?
    n_scen = len(chi_data.keys())
    
    # Load in population data at 2.5 arc minute resolution (~ 4.6 km at the equator)
    print('Reading in population data')
    var_name = 'UN WPP-Adjusted Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'
    ds_GPW = netCDF4.Dataset(os.path.join(GPW_dir,'gpw_v4_population_density_adjusted_rev11_2pt5_min.nc'),'r')
    hrz_grid_GPW = latlontools.gen_grid(lon_stride=360/8640,lat_stride=180/4320,half_polar=False,center_180=False)
    cid_grid = (np.flip(ds_GPW[var_name][10,:,:],axis=0)).astype(np.int)
    # Convert from people per m2 of land to just "people" (NB: entry 3 is the GPWv4 2015 estimate)
    pop_grid  = np.double(np.flip(ds_GPW[var_name][3,:,:],axis=0)).filled(0.0) * 1.0e-6 # Convert from ppl/km2 to ppl/m2
    land_grid = np.double(np.flip(ds_GPW[var_name][8,:,:],axis=0)).filled(0.0) * 1.0e6  # Convert from km2 to m2
    pop_grid *= land_grid
    # Need the country names
    cid_table = pandas.read_csv(os.path.join(GPW_dir,'gpw_v4_national_identifier_grid_rev11_lookup.txt'),delimiter='\t')
    
    # Grid size
    ny = pop_grid.shape[0]
    nx = pop_grid.shape[1]
    
    # Read in the mapping data associating each country with a WHO member state or region
    GPW_to_WHO = pickle.load(open(os.path.join(mort_preprocessor_dir,'GPW_to_WHO.pkl'),'rb'))
    
    # Load in WHO data (population and total mortalities for each country, age, and disease - unisex)
    print('Reading mortality data')
    f_country = os.path.join(WHO_dir,'GHE2016_Deaths_2016-country.xls')
    f_region  = os.path.join(WHO_dir,'GHE2016_Deaths_WHOReg_2000_2016.xls')
    who_data = read_who_data(who_by_country_xls=f_country,who_by_region_xls=f_region,as_rates=False)
    who_age_ceilings = []
    for i_age, age_range in enumerate(who_data.keys()):
        # WHO sheets take the form 'A-B' ('0-4', '5-14', and so on)
        # Therefore the upper bound is actually B+1. The exception
        # is for the last
        if age_range[-1] == '+' or i_age == (len(who_data.keys())-1):
            max_age = np.inf
        else:
            max_age = int(age_range.split('-')[-1])
        who_age_ceilings.append(max_age)
    
    # Conservatively regrid the concentration maps to the population grid
    # Also take this opportunity to generate the output array
    if verbose:
        print('Regridding exposure data to population grid')
    pop_to_chi_regrid = None
    for scen in chi_data.keys():
        if pop_to_chi_regrid is None or not common_chi_grid:
            pop_to_chi_regrid = regrid.gen_regridder(chi_data[scen]['grid'],hrz_grid_GPW)
        for ef in chi_data[scen].keys():
            if ef == 'grid':
                continue
            data_used = False
            for ERF in ERF_set:
                data_used = data_used or ef == ERF.exposure_factor
            chi_data[scen][ef] = pop_to_chi_regrid(chi_data[scen][ef])
    
    # Data to be stored: baseline and policy mortality for each ERF and each country
    n_erf = len(ERF_set)
    
    # For now, assume one random parameter per ERF
    n_rng_params = n_erf
    
    unique_cids = cid_table['Value'].values
    n_country = len(unique_cids)
    country_names = cid_table['NAME0'].values
    
    valid_grid = np.logical_and(pop_grid > 0,cid_grid <= 999)
    total_pop = np.nansum(pop_grid) 
    missing_pop = np.nansum(pop_grid[np.logical_not(valid_grid)])
    #print('Total missing population: {:.0f} of {:.0f} {:0.9%}'.format(missing_pop,total_pop,missing_pop/total_pop))         

    if US_states:
        state_mask_nc = netCDF4.Dataset(state_mask_file,'r')
        try:
            state_list = [x for x in state_mask_nc.variables.keys() if x not in state_mask_nc.dimensions]
            n_states = len(state_list)
            regionid_grid = np.zeros(cid_grid.shape,np.int)
            regionid_list = [[]] * n_country
            region_names = [[]] * n_country
            i_US = country_names.tolist().index('United States of America')
            regionid_list[i_US] = list(range(n_states))
            region_names[i_US] = state_list
            r_3D = np.zeros([n_states] + list(cid_grid.shape))
            for i_state, state in enumerate(state_list):
                r_3D[i_state,...] = state_mask_nc[state][...]
            regionid_grid = np.argmax(r_3D,axis=0)
            regionid_grid[np.sum(r_3D,axis=0) < 1.0e-20] = 999
        finally:
            state_mask_nc.close()
    else:
        regionid_grid = np.zeros((1,1),np.int)
        regionid_list = None
        region_names  = None
   
    if verbose: 
        print('Initiating Monte-Carlo step with {:d} draws on {:d} parameters'.format(n_rand,n_rng_params))
    
    # Make sure the ERFs at least have unique names
    name_list = [ERF.name for ERF in ERF_set]
    assert len(np.unique(name_list)) == len(name_list), 'Each meta-ERF must have a unique name'
     
    # Generate parameter dictionary for mcstats
    # mcstats needs handles for all the parameter generators
    param_dict = {}
    for ERF in ERF_set:
        for i_param, p_g in enumerate(ERF.param_gen):
            param_dict[ERF.name + '_param{:d}'.format(i_param)] = ERF
    
    # ERF_set links to the same objects as param_dict, so can ignore p_d in the call
    targ_fn = lambda p_d, i_batch, batch_size : run_sim_batch(ERF_set, country_names, who_age_ceilings, pop_grid, cid_grid, unique_cids,
                                                              who_data, GPW_to_WHO, chi_data, ref_map, i_batch=i_batch, n_per_batch=batch_size,
                                                              n_draws=n_rand, regionid_grid=regionid_grid,regionid_list=regionid_list,
                                                              region_names=region_names,verbose=True,n_cpus=n_cpus)
    output_list, timing_data = mcstats.run_mc(targ_fn, param_dict, n_rand, rng_opts=rng_opts,
                                 batch_size=n_rand, unpack_batch=False)
    output_data = output_list[0]

    return output_data

# Internal function used by mcstats
def run_sim_batch(ERF_set, country_names, who_age_ceilings, pop_grid_2d, cid_grid_2d, unique_cids, who_data, GPW_to_WHO, chi_data, ref_map,
                  i_batch, n_per_batch, n_draws, regionid_grid=None, regionid_list=None, region_names=None, verbose=True, n_cpus=None):
    output_data = {}
    output_data['aux'] = {'countries': country_names}
    n_country = len(unique_cids)

    # Use this to avoid non-calculations
    pop_ok = np.logical_and(pop_grid_2d > 0,cid_grid_2d < 10000)
    n_ok = np.count_nonzero(pop_ok)

    pop_grid_dbl = pop_grid_2d[pop_ok].astype(np.double)
    pop_grid_dbl_R = multiprocessing.RawArray('d',pop_grid_dbl.size)
    np.copyto(np.frombuffer(pop_grid_dbl_R),pop_grid_dbl)

    # Drop to 1 if necessary
    if n_cpus is None:
        n_cpus = 1
        print('Assuming single CPU only')

    # Having terrible trouble with this
    cid_grid = cid_grid_2d[pop_ok]
    cid_grid_R = multiprocessing.RawArray('d',n_ok)
    np.copyto(np.frombuffer(cid_grid_R),cid_grid.astype(np.double))

    # For subregion calculations using country-wide statistics
    get_regions = regionid_list is not None
    if get_regions:
        regionid_grid_R = multiprocessing.RawArray('d',n_ok)
        np.copyto(np.frombuffer(regionid_grid_R),regionid_grid[pop_ok].astype(np.double))
        n_region = 0
        region_names_condensed = []
        for i_country, r_list in enumerate(regionid_list):
            n_region += len(r_list)
            region_names_condensed += region_names[i_country][:]
        assert len(region_names_condensed) == n_region, 'Mismatch in region count: {:d} vs {:d}'.format(
                n_region, len(region_names_condensed))
    else:
        n_region = 0
        regionid_list   = None
        region_names    = None
        regionid_grid_R = None
        region_names_condensed = None
    output_data['aux']['regions'] = region_names_condensed

    if verbose:
        print('Beginning outer loop of batch {:d} of {:d}'.format(i_batch,int(np.ceil(n_draws/n_per_batch))))

    n_calc_total = len(ERF_set) * (len(chi_data.keys()) - 1)
    with tqdm(total=n_calc_total) as pbar:
        for ERF in ERF_set:
            # Which exposure factor does this ERF deal with?
            ef = ERF.exposure_factor

            # Update the ERF parameters
            ERF.update_erf_params()
 
            # Generate output
            if ef not in output_data.keys():
                output_data[ef] = {}
            output_data[ef][ERF.name] = {}
        
            # Extract the indices of the GHE data which are relevant
            pop_idx = 0
            disease_idx = ERF.disease_data.id_GHEs(who_data)

            # Calculate mortality by country for the target disease(s), covering the target age range
            ref_morts = np.zeros(n_country)
            ref_pop   = np.zeros(n_country)
            # First figure out the fractions of the WHO age brackets covered by this ERF
            # NB: We need finite age ranges, so treat infinite values as being = 100
            min_age_ERF = ERF.min_age
            max_age_ERF = min(100,ERF.max_age+1)
            max_age_WHO = 0
            age_fracs = np.zeros(len(who_age_ceilings))
            for i_WHO in range(len(who_age_ceilings)):
                # Age ranges are given as integers and are inclusive
                # Convert to "floats" (ie min <= age < max)
                min_age_WHO = np.double(max_age_WHO)
                max_age_WHO = min(100,np.double(who_age_ceilings[i_WHO] + 1))
                if max_age_WHO <= min_age_ERF or min_age_WHO >= max_age_ERF:
                   bin_frac = 0.0
                else:
                   # Some part of the WHO age bin is within the ERF age bin
                   inner_range = (min(max_age_WHO,max_age_ERF) - max(min_age_WHO,min_age_ERF))
                   outer_range = max_age_WHO - min_age_WHO
                   bin_frac = (inner_range / outer_range)
                age_fracs[i_WHO] = bin_frac
        
            # Get the total mortalities in this age range for this disease, for each country, and estimate the mort rate
            # Start with a map of baseline incidence (rate)
            baseline_incidence_rate = np.zeros(n_ok) + np.nan
            #inc_by_country = np.zeros(n_country)
            # For now, ignore population growth
            pop_inflator = np.zeros(n_country) + 1.0
            ref_ofage    = np.zeros(n_country)
            for i_country in range(n_country):
                cid = unique_cids[i_country]
                cid_mask = cid_grid == unique_cids[i_country]
                mort = 0.0
                aff_pop = 0.0
                all_pop = 0.0
                # Identify the WHO entry associated with this CID
                WHO_name = GPW_to_WHO[cid]
                age_min = 0
                for i_age_bin, age_max in enumerate(who_age_ceilings):
                    # Must include ALL age bins, to get the right fractions
                    if np.isinf(age_max):
                        sheet_name = '{:d}+'.format(age_min)
                    else:
                        sheet_name = '{:d}-{:d}'.format(age_min,age_max)
                    who_df = who_data[sheet_name]
                    if 'R_' in WHO_name:
                        country_df = who_df[WHO_name[2:]]
                    else:
                        country_df = who_df[WHO_name]
                    aff_pop  += age_fracs[i_age_bin] * country_df[pop_idx]
                    all_pop  += country_df[pop_idx]
                    mort     += age_fracs[i_age_bin] * np.nansum(country_df[disease_idx])
                    age_min = age_max + 1
                # Store long-term
                ref_morts[i_country] = mort    # << NB: This will be weird for countries using proxies
                ref_pop[  i_country] = aff_pop # << NB: This will be weird for countries using proxies
                ref_ofage[i_country] = aff_pop/all_pop
                # This is the incidence rate *among the ENTIRE population*
                baseline_incidence_rate[cid_mask] = (mort/all_pop)

            output_data[ef][ERF.name]['ref_morts'] = ref_morts
            output_data[ef][ERF.name]['ref_pop']   = ref_pop
            output_data[ef][ERF.name]['age_frc']   = ref_ofage
        
            # Produce a map of the total baseline incidence, using the GPW population grid
            # This is equal to [affected fraction] * [total population] * [mortality rate in affected population]
            baseline_incidence = baseline_incidence_rate * pop_grid_dbl
   
            # Transform the reference concentration
            chi_ref_raw = chi_data[ref_map][ef][pop_ok]
            chi_ref     = ERF.erf.process_chi(chi_ref_raw)

            # Copy to shared arrays
            baseline_incidence_R = multiprocessing.RawArray('d',n_ok)
            np.copyto(np.frombuffer(baseline_incidence_R),baseline_incidence)
            chi_ref_raw_R = multiprocessing.RawArray('d',n_ok)
            np.copyto(np.frombuffer(chi_ref_raw_R),chi_ref_raw)
            chi_ref_R = multiprocessing.RawArray('d',n_ok)
            np.copyto(np.frombuffer(chi_ref_R),chi_ref)
            baseline_incidence_R = multiprocessing.RawArray('d',n_ok)
            np.copyto(np.frombuffer(baseline_incidence_R),baseline_incidence)

            for scen_name, scen_data in chi_data.items():
                if scen_name == ref_map:
                    continue

                output_data[ef][ERF.name][scen_name] = {}
                chi_scen_raw = scen_data[ef][pop_ok]
                chi_scen     = ERF.erf.process_chi(chi_scen_raw)
                chi_delta_raw = (chi_scen_raw - chi_ref_raw)

                chi_scen_R = multiprocessing.RawArray('d',n_ok)
                np.copyto(np.frombuffer(chi_scen_R),chi_scen)
                chi_delta_raw_R = multiprocessing.RawArray('d',n_ok)
                np.copyto(np.frombuffer(chi_delta_raw_R),chi_delta_raw)
         
                # Calculate the relative risk map for this RR
                batch_morts  = np.zeros((n_country,n_per_batch)) + np.nan
                batch_rmorts = np.zeros((n_region,n_per_batch)) + np.nan

                extra_data = {}

                # Things to tell each worker (all common)
                init_args = (i_batch,n_per_batch,n_draws,n_country,ERF,pop_inflator,unique_cids,
                             chi_scen_R, chi_ref_R, chi_ref_raw_R, baseline_incidence_R, pop_grid_dbl_R,
                             chi_delta_raw_R, cid_grid_R, regionid_grid_R, regionid_list, n_region)
                             #chi_delta_raw_R, cid_precalc_int_R)

                with multiprocessing.Pool(processes=n_cpus, initializer=init_worker, initargs=init_args) as mp_pool:
                    result_list = mp_pool.map(worker_calc, range(n_per_batch))
                    for result_temp in result_list:
                        morts_temp  = result_temp[0]
                        rmorts_temp = result_temp[1]
                        extra_temp  = result_temp[2]
                        i           = result_temp[3]
                        if extra_temp is not None:
                            extra_data = extra_temp
                        if i is not None:
                            batch_morts[:,i]  = morts_temp[:]
                            batch_rmorts[:,i] = rmorts_temp[:]

                if use_tqdm:
                    pbar.update(1)

                output_data[ef][ERF.name][scen_name]['morts_vs_ref']        = batch_morts.copy()
                output_data[ef][ERF.name][scen_name]['region_morts_vs_ref'] = batch_rmorts.copy()
                output_data[ef][ERF.name][scen_name]['delta_exp'   ] = extra_data['delta_exp'].copy()
                output_data[ef][ERF.name][scen_name]['base_exp'    ] = extra_data['base_exp'].copy()
                output_data[ef][ERF.name]['base_morts'  ]            = extra_data['base_morts'].copy()
                output_data[ef][ERF.name]['affected_pop']            = extra_data['affected_pop'].copy()

    return output_data
