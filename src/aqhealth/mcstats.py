import numpy as np
import scipy.stats
import time

def iter_tri(aNew,lCB,cEst,uCB,CI):
    hNew = (1.0-CI) * (cEst-aNew)/(np.power(lCB-aNew,2.0))
    # Quadratic
    pCo = np.zeros(3)
    pCo[0] = hNew
    pCo[1] = CI - 1 - 2*uCB*hNew
    pCo[2] = hNew*uCB*uCB + cEst*(1.0-CI)
    qRoot = np.roots(pCo)
    qRoot[np.abs(np.imag(qRoot)) > 0.0] = np.nan
    qRoot[np.logical_not(np.isfinite(qRoot))] = np.nan
    qRoot[qRoot<=uCB] = np.nan
    qRoot = qRoot[np.logical_not(np.isnan(qRoot))]
    if qRoot.size > 1:
        bNew = np.min(qRoot)
    else:
        bNew = qRoot[0]
    
    # Test against total area
    zDist = np.abs(hNew*(bNew-aNew)/2.0 - 1.0)

    return zDist, aNew, bNew, hNew

# Function to find actual triangular distribution bounds, given the 95% CIs
def iter_tri_bounds(lower_cb,mid_val,upper_cb,ci_pcg):
    # Initial guess
    ci = ci_pcg / 100.0
    x0 = lower_cb - (ci_pcg/2.0)*(mid_val - lower_cb)
    assert x0 < lower_cb, 'Cannot iterate on right triangle'
    iter_fn = lambda x, lcb=lower_cb,mv=mid_val,ucb=upper_cb,cifix=ci : iter_tri(x, lcb, mv, ucb, cifix)[0]
    min_bound=lower_cb - (2.0*(mid_val-lower_cb))
    max_bound=lower_cb
    fmin_out = scipy.optimize.fminbound(iter_fn,min_bound,max_bound)
    x_final = fmin_out
    ignore, lower_bound, upper_bound, h = iter_tri(x_final, lower_cb, mid_val, upper_cb, ci)
    return lower_bound, upper_bound, h

# Function to model uniformly-distributed variable given a variate
# Accepts a vector of random values
def uni_dist(rand,low,high):
    return low + (high-low)*rand

# Function to model triangularly-distributed variable given a variate
# Accepts a vector of random values
def tri_dist(rand,low,mode,high):
    rand_arra = np.array(rand)
    mid_rand = (mode-low)/(high-low)
    out_val = np.where(rand < mid_rand,low + np.sqrt(rand*(high-low)*(mode-low)),
                       high - np.sqrt((1.0-rand)*(high-low)*(high-mode)))
    return out_val

# Function to model normally-distributed variable given a variate
# Accepts a vector of random values
def norm_dist(rand,mean,sdev):
    # Standard normal distribution
    norm_val = scipy.stats.norm.ppf(rand)
    out_val = mean + (norm_val*sdev) 
    return out_val

# Random variable class. This class (and any subclass) must:
# 1. Be initialized with nothing but the relevant parameters
# 2. On initialization, calculate and store the support of their distribution
# 3. Have two additional functions:
#     --> generate: A function which accepts a vector of N variates between 0 and 1, 
#                   and stores the appropriate samples from the target distribution
#                   internally IF NECESSARY
#     --> provide:  A function which accepts a draw index (<= N) and returns the 
#                   drawn value from an earlier "generate" call
class rand_var:
    # Most basic random variable class - just spits back the input array
    def __init__(self):
        self.draws = None
        # No support (cannot generate)
        self.support = [+np.inf,-np.inf]
    
    def generate(self,rand_vec):
        raise ValueError('Not a usable class')
    
    def provide(self,i_draw=None):
        if i_draw is None:
            return self.draws
        else:
            return self.draws[i_draw]
        
    def safe_generate(self,rand_vec):
        assert np.max(rand_vec) <= 1.0 and np.min(rand_vec) >= 0.0, 'Not in valid range'
        assert self.support[0] >= self.support[1], 'Support not valid'
        self.generate(rand_vec)

class fixed_var(rand_var):
    # Non-random variable
    def __init__(self,fixed_val):
        self.val = fixed_val
        self.support = [fixed_val,fixed_val]
    
    def generate(self,rand_vec):
        # Don't bother storing the data - use overridden "provide" instead
        return
    
    def provide(self,i_draw):
        return self.val

class uni_var(rand_var):
    # Uniformly-distributed random variable
    def __init__(self,low=0.0,high=1.0):
        self.low  = low
        self.high = high
        self.support = [low,high]
    
    def generate(self,rand_vec):
        self.draws = uni_dist(rand_vec,low=self.low,high=self.high)

class tri_var(rand_var):
    # Triangularly-distributed random variable
    def __init__(self,low=0.0,mid=0.5,high=1.0):
        self.low  = low
        self.mid  = mid
        self.high = high
        self.support = [low,high]
        self.draws = None
    def generate(self,rand_vec):
        self.draws = tri_dist(rand_vec,low=self.low,mode=self.mid,high=self.high)

class norm_var(rand_var):
    # Normally-distributed random variable
    def __init__(self,mean=0.0,sdev=1.0):
        self.mean = mean
        self.sdev = sdev
        self.support = [-np.inf,+np.inf]
        self.draws = None
    
    def generate(self,rand_vec):
        self.draws = norm_dist(rand_vec,mean=self.mean,sdev=self.sdev)

class discrete_var(rand_var):
    # Variable with discrete values
    def __init__(self,value_list,probability_list=None):
        n_values = len(value_list)
        if probability_list is None:
            self.p_list = [1/n_values] * n_values
        else:
            assert len(probability_list) == n_values, 'If probabilities are given, need one per entry'
            # Assert that the total probability is 1
            assert np.isclose(np.sum(probability_list),1.0,atol=1e-10,rtol=1e-12),'Probability sum must equal 1 for discrete distribution'
            # Copy the list
            self.p_list = probability_list
        p_max = 0.0
        self.p_threshold = np.cumsum(self.p_list)
        self.p_threshold[-1] = 1.0
        self.value_list = value_list
        # If all elements are simple scalars, we can skip a step during generation
        self.fast_draw = np.all([np.isscalar(x) for x in value_list])
    
    def generate(self,rand_vec):
        self.draws = np.zeros(len(rand_vec))
        condlist = [rand_vec <= x for x in self.p_threshold]
        if self.fast_draw:
            # The below works fine when everything is scalar
            self.draws = np.select(condlist,self.value_list)
        else:
            # The below is robust, but likely slower
            draw_idx = np.select(condlist,list(range(len(self.value_list))))
            self.draws = []
            for idx in draw_idx:
                self.draws.append(self.value_list[idx])

def create_pqrng(n_params,rng_name='pseudorandom',random_seed=None,sobol_skip=True):
    if rng_name == 'pseudorandom':
        # If no seed given, use current time
        if random_seed is None:
            random_seed = int(time.time())
        # Create the random number generator
        rng = np.random.RandomState(random_seed)
        rand_generator = rng.rand
    elif rng_name == 'sobol':
        import SALib.sample.sobol_sequence
        # Skip initial values; some authors recommend skipping P, where P is the largest
        # power of 2 smaller than the number of draws
        if sobol_skip:
            skip_factor = 1
        else:
            skip_factor = 0
        n_skip = lambda n_draws : skip_factor*int(np.power(2,np.floor(np.log2(n_draws))))
        rand_generator = lambda n_draws, n_params : SALib.sample.sobol_sequence.sample(
            n_draws+n_skip(n_draws),n_params)[n_skip(n_draws):,:]
    elif rng_name == 'saltelli':
        import SALib.sample.saltelli
        raise ValueError('Saltelli sequence not yet functional')
        problem = {
          'num_vars': n_params,
          'names': list(range(n_params)),
          'bounds': [[0, 1]]*n_params
        }
        rand_generator = lambda n_draws, n_params : saltelli.sample(problem, n_draws)
    else:
        raise ValueError('Invalid RNG name')
    return rand_generator
        
def run_mc(targ_fn,param_dict,n_draws,rand_generator=None,verbose=False,rng_opts={},batch_size=1,unpack_batch=False):
    
    batch_run = batch_size > 1
    if batch_run:
        n_batch = int(np.ceil(n_draws/batch_size))
    
    t_start = time.time()
    t_now = t_start
    timing_data = {}
    
    # How many parameters do we have?
    n_params = len(param_dict.keys())
    
    # Default to a simple pseudorandom generator, but allow option pass-through
    if rand_generator is None:
        rand_generator = create_pqrng(n_params,**rng_opts)
    t_last = t_now
    t_now = time.time()
    timing_data['rng_generate'] = t_now - t_last
        
    # Draw variables between 0 and 1 (uniform)
    uni_data = rand_generator(n_draws,n_params)
    t_last = t_now
    t_now = time.time()
    timing_data['uni_draw'] = t_now - t_last
    
    # Convert to the target distributions
    i_param = 0
    for param_name, param_obj in param_dict.items():
        param_obj.generate(uni_data[:n_draws,i_param])
        i_param += 1
    t_last = t_now
    t_now = time.time()
    timing_data['var_convert'] = t_now - t_last
        
    # Run the Monte-Carlo code
    if batch_run:
        output_data = []
        # Parameter object handling is performed inside the function
        for i_batch in range(n_batch):
            output_batch = targ_fn(param_dict,i_batch,batch_size)
            # If "unpack_batch", results are returned as a list to
            # make it easier to append. Otherwise, we just append
            # each batch entry
            if unpack_batch:
                output_data += output_batch[:]
            else:
                output_data.append(output_batch)
    else:
        output_data = []
        for i_draw in range(n_draws):
            draw_data = {}
            for param_name, param_obj in param_dict.items():
                draw_data[param_name] = param_obj.provide(i_draw)
            output_data.append(targ_fn(**draw_data))
    t_last = t_now
    t_now = time.time()
    timing_data['run_mc'] = t_now - t_last
    return output_data, timing_data

def test_mc(N_samples=1000,plot_outputs=True):
    if plot_outputs:
        try:
            import matplotlib.pyplot as plt
        except:
            plot_outputs = False
            print('Cannot run matplotlib. Only text output will be created')
    
    # Sanitize..
    N = int(N_samples)
    
    # Make a function which takes 20 parameters and adds them (test central limit theorem)
    param_dict = {}
    for i_param in range(20):
        param_dict['p' + str(i_param)] = uni_var(low=-0.5,high=+0.5)

    # Sum of expected variances - note variance of uniform dist over an interval of [0,1] is 1/12
    snsq = 20 / 12
    sn = np.sqrt(snsq)

    # Function to add variables
    def mc_sub_fn(**kwargs):
        return sum(kwargs.values())/np.sqrt(len(kwargs.values())/12)
    #def mc_sub_fn(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19):
    #    return (p0+p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13+p14+p15+p16+p17+p18+p19) / sn

    # Two sets of output
    set_name_list = ['pseudorandom','sobol']
    if plot_outputs:
        f, ax_var = plt.subplots(1,1,figsize=(8,5))
        f, ax_hist = plt.subplots(1,len(set_name_list),figsize=(8,5))
        bin_edge = np.arange(-5,5,0.1)
    
    # Run two sets of tests - once with PRNG, once with QRNG
    for i_set, set_name in enumerate(set_name_list):
        t_start = time.time()
        # Generate (P/Q)RNG separately
        rng = create_pqrng(len(param_dict.keys()),rng_name=set_name,sobol_skip=True)
        out_data, timing_data = run_mc(mc_sub_fn,param_dict,N,rand_generator=rng)
        # One-shot..
        #out_data, ignore = run_mc(mc_sub_fn,param_dict,N,rand_generator=None,rng_opts={'rng_name': set_name, 'sobol_skip': True})
        t_end = time.time()
        
        print('Data calculated for {:s} with {:d} samples. Timing:'.format(set_name,N))
        for key, item in timing_data.items():
            print('--> {:15s}: {:5.2f}s'.format(key,item))

        #sample_pts = np.linspace(10,N,100)
        #sample_pts = np.logspace(start=0,stop=np.log10(N),num=100,base=10)
        sample_pts = np.geomspace(start=10,stop=N,num=200)
        result_var = np.zeros(len(sample_pts))
        for i_N, targ_N in enumerate(sample_pts):
            result_var[i_N] = np.var(out_data[:int(targ_N)])
            
        # Show results
        if plot_outputs:
            ax = ax_hist[i_set]
            ax.hist(out_data,bin_edge)
            ax.set_xlabel('Estimated result')
            ax.set_title('{:s}'.format(set_name))
            ax.set_ylabel('Frequency')

            ax_var.plot(sample_pts,result_var,label=set_name,marker='.')
           
        print('Expected vs simulated variance for case {:20s}: {:7.3f} vs {:7.3f}'.format(
            set_name,1.0,result_var[-1]))
    if plot_outputs:
        # Expected variance
        ax_var.plot([0,N],[1.0,1.0],color='k',linestyle='--',label='True result')
        ax_var.legend()
        ax_var.set_xlabel('Number of samples used N')
        ax_var.set_ylabel('Variance of results 1:N')
        ax_var.set_xscale('log')
        #ax_var.set_xlim(0,N)
        ax_var.set_xlim(10,N)
    return None
