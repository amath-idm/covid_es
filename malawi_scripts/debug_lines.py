'''
Run a sweep across parameters -- like sweep_grid but do 1D sweeps instead of
grid sweeps.
'''

import sciris as sc
import covasim_es as ces

# Define general settings for the run
label   = 'debug_line' # The name of the experiment to run
verbose = 1 # Whether to print out progress along the way

# Handle how many parameter sets to run
n  = 10
n_cpus = 20
do_plot = True

def make_scenario():

    # Scenario sweep parameters (lower and upper bounds)
    scenario = sc.objdict(

        # ES parameters
        es_prev_thresh  = [0.003], # Minimum prevalence threshold for ES detection
        es_sensitivity  = [0.084], # Sensitivity of ES to detection once above threshold
        es_specificity  = [1.0], # Specificity of ES detection
        es_frequency    = [3], # How frequently samples are taken
        es_n_samples    = [9], # How many samples are taken each time
        es_trigger      = [1], # The trigger rate for ES detections (of a new variant)

        # Intervention parameters
        clin_test_prob   = [0.0004], # Per-day probability of a person with symptoms testing
        clin_trigger     = [30], # The trigger rate for clinical detections
        intervention_eff = [0.3], # The efficacy of the lockdown/mask/distancing intervention

        # Variant/immunity parameters
        variant_trans   = [1.0], # Relative transmissibility of the new variant compared to wild type
        variant_sev     = [2.0], # Severity of the new variant compared to wild type
        baseline_imm    = [0.0]  # Fraction of the population that has prior immunity to COVID
    )

    return scenario

# Simulation parameters
sim_pars = sc.objdict(
    location  = 'malawi',
    pop_size  = 200_000, # Population size
    pop_scale = 5, # Total population 1.0m
    pop_infected = 20,
    rescale   = False,
    start_day = '2022-06-01', # First day of simulation
    end_day   = '2022-12-01', # Last day of simulation
    verbose   = -1, # How much detail to print (0 = none; -1 = one line per sim)
    rand_seed = 0, # Random seed offset
)

def printmean(data):
    import numpy as np
    data = np.array(data)
    mean = data.mean()
    std = data.std()
    string = f'{mean:n} ± {sc.sigfig(std, 2)}'
    return string


if __name__ == '__main__':

    scenario = make_scenario()
    
    # Create, run, and export the experiment
    exp = ces.Experiment(label=label, n=n, scenario=scenario, method='line', sim_pars=sim_pars, verbose=verbose)
    exp.run(n_cpus=n_cpus)
    exp.export()

    # Analyze results
    print('Results:')
    print(exp.results)
    df = exp.results
    es0 = df[df.use_es==0]
    es1 = df[df.use_es==1]
    dalys_es0 = es0.dalys.mean()
    dalys_es1 = df[df.use_es==1].dalys.mean()
    print('DALYs:', printmean(es0.dalys), printmean(es1.dalys))
    print('Days active:', printmean(es0.beta_days), printmean(es1.beta_days))
    print('Start day:', printmean(es0.beta_start), printmean(es1.beta_start))

    # Plot
    if do_plot:
        import covasim as cv
        sims_es0 = [sim for sim in exp.msim.sims if sim.scen.use_es==0]
        sims_es1 = [sim for sim in exp.msim.sims if sim.scen.use_es==1]
        msim_es0 = cv.MultiSim(sims=sims_es0)
        msim_es1 = cv.MultiSim(sims=sims_es1)
        msim_es0.mean()
        msim_es1.mean()
        msim = cv.MultiSim.merge(msim_es0, msim_es1, base=True)
        msim.plot('overview');
