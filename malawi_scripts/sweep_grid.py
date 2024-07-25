'''
Run a sweep across parameters.
'''

import sciris as sc
import covasim_es as ces

# Define general settings for the run
n       = 20 # How many simulations to run
label   = 'threshold_range' # The name of the experiment to run
debug   = 0 # Whether to run a minimal set of sims
verbose = 0 # Whether to print out progress along the way

# Handle how many parameter sets to run
if debug:
    n  = 2
    n_cpus = 6
    do_plot = True
else:
    n = n
    n_cpus = None
    do_plot = False

def make_scenario(debug):

    # Scenario sweep parameters (lower and upper bounds)
    scenario = sc.objdict(

        # ES parameters
        es_prev_thresh  = [[0.001, 0.02], [0.001]][debug], # Minimum prevalence threshold for ES detection
        es_sensitivity  = [[0.25, 1.0], [1.0]][debug], # Sensitivity of ES to detection once above threshold
        es_specificity  = [1.0], # Specificity of ES detection
        es_frequency    = [[1, 14], [14]][debug], # How frequently samples are taken
        es_n_samples    = [[1, 10], [5]][debug], # How many samples are taken each time
        es_trigger      = [2, 10], # The trigger rate for ES detections (of a new variant)

        # Intervention parameters
        clin_test_prob   = [[0.0001, 0.001, 0.01], [0.001]][debug], # Per-day probability of a person with symptoms testing
        clin_trigger     = [2, 10], # The trigger rate for clinical detections
        intervention_eff = [[0.1, 0.3], [0.3]][debug], # The efficacy of the lockdown/mask/distancing intervention

        # Variant/immunity parameters
        variant_trans   = [4.0], # Relative transmissibility of the new variant compared to wild type
        variant_sev     = [2.0], # Severity of the new variant compared to wild type
        baseline_imm    = [0.15]  # Fraction of the population that has prior immunity to COVID
    )

    return scenario

# Simulation parameters
sim_pars = sc.objdict(
    location  = 'malawi',
    pop_size  = 100_000, # Population size
    pop_scale = 10, # Total population 1.0m
    pop_infected = 200,
    rescale   = False,
    start_day = '2022-06-01', # First day of simulation
    end_day   = '2022-12-01', # Last day of simulation
    verbose   = -1 # How much detail to print (0 = none; -1 = one line per sim)
)

if __name__ == '__main__':

    scenario = make_scenario(debug)
    
    # Create, run, and export the experiment
    exp = ces.Experiment(label=label, n=n, scenario=scenario, sim_pars=sim_pars, verbose=verbose)
    exp.run(n_cpus=n_cpus)
    exp.export()

    # Analyze results
    print('Results:')
    print(exp.results)
    df = exp.results

    # Plot
    if do_plot:
        exp.msim.plot('overview')
