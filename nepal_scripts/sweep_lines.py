'''
Run a sweep across parameters -- like sweep_grid but do 1D sweeps instead of
grid sweeps.
'''

import sciris as sc
import covasim_es as ces

# Define general settings for the run
n       = 1000 # How many simulations to run
label   = 'nepal_oct23' # The name of the experiment to run
debug   = 0 # Whether to run a minimal set of sims
verbose = 0 # Whether to print out progress along the way

# Handle how many parameter sets to run
if debug:
    n = 2
    ncpus = 10
    do_plot = True
else:
    n = n
    ncpus = 120 # For running in parallel on VMs
    do_plot = False


def make_scenario(debug=debug, default=False):

    # Scenario sweep parameters (lower and upper bounds)
    
    if debug:
        scenario = sc.objdict(
    
            # ES parameters
            es_prev_thresh  = [0.0035, [0.00035, 0.03]], # Minimum prevalence threshold for ES detection
            es_sensitivity  = [0.084], # Sensitivity of ES to detection once above threshold
            es_specificity  = [1.0], # Specificity of ES detection
            es_frequency    = [3], # How frequently samples are taken
            es_n_samples    = [9], # How many samples are taken each time
            es_trigger      = [1], # The trigger rate for ES detections (of a new variant)
            es_lag          = [4], # The number of days from wastewater collection to action
    
            # Intervention parameters
            clin_test_prob   = [0.0003, [0.0, 0.01]], # Per-day probability of a person with symptoms testing
            clin_trigger     = [30, [5, 100]], # The trigger rate for clinical detections
            intervention_eff = [0.2, [0.1, 0.3]], # The efficacy of the lockdown/mask/distancing intervention
    
            # Variant/immunity parameters
            variant_trans   = [1.0], # Relative transmissibility of the new variant compared to wild type
            variant_sev     = [1.0], # Severity of the new variant compared to wild type
            baseline_imm    = [0.0]  # Fraction of the population that has prior immunity to COVID
        )
    
    else:
        scenario = sc.objdict(
    
            # ES parameters
            es_prev_thresh  = [0.0035, [0.00035, 0.01, 0.03]], # Minimum prevalence threshold for ES detection
            es_sensitivity  = [0.084, [0.04, 1.0]], # Sensitivity of ES to detection once above threshold
            es_specificity  = [1.0], # Specificity of ES detection
            es_frequency    = [3, [1, 14]], # How frequently samples are taken
            es_n_samples    = [9, [1, 3, 100]], # How many samples are taken each time
            es_trigger      = [1, [2, 5]], # The trigger rate for ES detections (of a new variant)
            es_lag          = [4, [2, 10, 21]], # The number of days from wastewater collection to action
    
            # Intervention parameters
            clin_test_prob   = [0.0003, [0.0, 0.0001, 0.001, 0.003, 0.01]], # Per-day probability of a person with symptoms testing
            clin_trigger     = [30, [5, 10, 100]], # The trigger rate for clinical detections
            intervention_eff = [0.2, [0.1, 0.3]], # The efficacy of the lockdown/mask/distancing intervention
    
            # Variant/immunity parameters
            variant_trans   = [1.0], # Relative transmissibility of the new variant compared to wild type
            variant_sev     = [1.0], # Severity of the new variant compared to wild type
            baseline_imm    = [0.0]  # Fraction of the population that has prior immunity to COVID
        )
    
    if default:
        scenario = sc.objdict({k:[v[0]] for k,v in scenario.items()})

    return scenario

# Simulation parameters
sim_pars = sc.objdict(
    location  = 'nepal',
    pop_size  = 500_000, # Population size
    scaled_pop = 2.734802e6, # Total population 2.7m
    pop_infected = 1000,
    rescale   = False,
    start_day = '2022-06-01', # First day of simulation
    end_day   = '2022-12-01', # Last day of simulation
    verbose   = -1, # How much detail to print (0 = none; -1 = one line per sim)
    rand_seed = 3847363, # Random seed offset
)

if __name__ == '__main__':

    scenario = make_scenario()
    
    # Create, run, and export the experiment
    exp = ces.Experiment(label=label, n=n, scenario=scenario, method='line', sim_pars=sim_pars, verbose=verbose)
    exp.run(ncpus=ncpus)
    exp.export()

    # Analyze results
    print('Results:')
    print(exp.results)
    df = exp.results

    # Plot
    # if do_plot:
    #     exp.msim.plot('overview')
