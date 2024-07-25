'''
Run a sweep across parameters.
'''

import sciris as sc
import covasim_es as ces

# Define general settings for the run
n     = 100 # How many simulations to run
label = 'example' # The name of the experiment to run
debug = 0 # Whether to run a minimal set of sims

# Handle how many parameter sets to run
if debug:
    n  = 2
    sl = slice(1)
    n_cpus = 6
    do_plot = True
else:
    n = n
    sl = slice(None)
    n_cpus = None
    do_plot = False


# Scenario sweep parameters (lower and upper bounds)
scenario = sc.objdict(

    # ES parameters
    es_prev_thresh  = [0.001, 0.02], # Minimum prevalence threshold for ES detection
    es_sensitivity  = [0.25, 1.0][sl], # Sensitivity of ES to detection once above threshold
    es_specificity  = [1.0], # Specificity of ES detection
    es_frequency    = [1, 14][sl], # How frequently samples are taken
    es_n_samples    = [1, 5][sl], # How many samples are taken each time
    es_trigger      = [1], # The trigger rate for ES detections (of a new variant)

    # Intervention parameters
    clin_test_prob   = [0.0001, 0.001, 0.01][sl], # Per-day probability of a person with symptoms testing
    clin_trigger     = [100], # The trigger rate for clinical detections
    intervention_eff = [0.1, 0.3][sl], # The efficacy of the lockdown/mask/distancing intervention

    # Variant/immunity parameters
    variant_trans   = [4.0], # Relative transmissibility of the new variant compared to wild type
    variant_sev     = [2.0], # Severity of the new variant compared to wild type
    baseline_imm    = [0.15]  # Fraction of the population that has prior immunity to COVID
)

# Simulation parameters
sim_pars = sc.objdict(
    location  = 'nepal',
    pop_size  = 100_000, # Population size
    pop_scale = 27.34802,
    pop_infected = 200,
    rescale   = False,
    start_day = '2022-06-01', # First day of simulation
    end_day   = '2022-12-01', # Last day of simulation
    verbose   = -1 # How much detail to print (0 = none; -1 = one line per sim)
)

# Create, run, and export the experiment
exp = ces.Experiment(label=label, n=n, scenario=scenario, sim_pars=sim_pars)
exp.run(n_cpus=n_cpus)
exp.export()

# Analyze results
print('Results:')
print(exp.results)
df = exp.results

# Plot
if do_plot:
    exp.msim.plot('overview')
