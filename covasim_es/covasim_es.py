"""
Define Covasim ES tools
"""

import os
import itertools as it
import numpy as np
import sciris as sc
import covasim as cv

__all__ = ['ES', 'beta_threshold', 'dalys', 'Experiment']


class ES(cv.Intervention):
    """
    Run an environmental surveillance program
    """

    def __init__(self, prev_thresh, sensitivity=1, specificity=1, frequency=1, n_samples=1, lag=4, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.thresh = prev_thresh
        self.sens = sensitivity
        self.spec = specificity
        self.verbose = verbose
        self.freq = round(frequency)
        self.n_samples = round(n_samples)
        self.lag = max(1, lag) # Can't have a lag of 0 since this is an intervention, results not updated yet
        self.detections = list()
        self.negatives = list()
        self.sample_dates = list()
        self.rng = np.random.RandomState(0)
        return


    def apply(self, sim):
        if not (sim.t % self.freq):
            t = sim.t - self.lag # Introduce a lag from the n_exposed to sample collection
            if t > 0:
                self.sample_dates.append(sim.t)
                self.detections.append(0)
                prev = sim.results['n_exposed'][t]/sim['pop_size']
                prob = np.maximum(0, (1-(1-1/np.exp(1))**((prev-self.thresh)/self.thresh))*self.sens)
                for i in range(self.n_samples):
                    if prev >= self.thresh:
                        detected = self.rng.rand() < prob
                        if detected:
                            if self.verbose:
                                print(f'ES detection: sim={sim.label}, day={sim.t}, prev={prev:0.4f}, thresh={self.thresh:0.4f}, prob={prob:0.4f}')
                            self.detections[-1] += 1
                self.negatives.append(self.n_samples - self.detections[-1])
        return



class beta_threshold(cv.Intervention):
    """
    Reduce beta after a certain level of detections
    """

    def __init__(self, change, clin_trigger=10, es_trigger=1, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.change = change
        self.clin_trigger = round(clin_trigger)
        self.es_trigger = round(es_trigger)
        self.verbose = verbose
        self.active = False
        self.orig_beta = None
        self.days_active = 0
        self.days = []
        return


    def apply(self, sim):

        es = sim.get_intervention(ES)
        if len(es.detections):
            es_detect = es.detections[-1]
            es_met = (es_detect >= self.es_trigger)
        else:
            es_detect = None
            es_met = False

        clin_detect = sim.results['new_diagnoses'][sim.t-1] * sim.rescale_vec[sim.t-1] # Unscaled results
        clin_met = (clin_detect >= self.clin_trigger)
        met = clin_met or es_met

        if self.active:
            self.days_active += 1
        if met and not self.active:
            self.orig_beta = sim['beta']
            sim['beta'] *= self.change
            self.days = [sim.t]
            self.active = True
            if self.verbose:
                print(f'On day {sim.t} for {sim.label}, changed beta to {sim["beta"]} because clinical={clin_met}, ES={es_met}')
        return


class dalys(cv.Analyzer):

    def __init__(self, max_age=84, short_covid=None, long_covid=None, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self.short_covid = short_covid if short_covid else dict(prop=0.7, dur=0.02, wt=0.2)
        self.long_covid  = long_covid  if long_covid  else dict(prop=0.1, permanent=0.001, dur=0.5, wt=0.05)
        return

    def __getitem__(self, key):
        return getattr(self, key)

    def initialize(self, sim):
        super().initialize()
        if sim['pop_scale'] > 1 and sim['rescale']:
            raise NotImplementedError('Analyzer not designed for dynamic rescaling')
        return

    def apply(self, sim):
        pass

    def finalize(self, sim):

        scale = sim['pop_scale']

        # Years of life lost
        dead = sim.people.dead
        years_left = np.maximum(0, self.max_age - sim.people.age)
        self.yll = (years_left*dead).sum()*scale
        self.deaths = dead.sum()*scale

        # Years lived with disability
        short = sc.objdict(self.short_covid)
        long  = sc.objdict(self.long_covid)
        inf = (~sim.people.naive)
        n_inf = inf.sum()*scale
        self.n_inf = n_inf
        self.yld_short = n_inf*short.prop*short.dur*short.wt
        self.yld_recov = n_inf*long.prop*long.dur*long.wt
        self.yld_permanent = (long.permanent*inf*years_left*long.wt).sum()*scale
        self.yld = self.yld_short + self.yld_recov + self.yld_permanent
        self.dalys = self.yll + self.yld
        return


class progress(cv.Analyzer):
    """ Save progress to a file """

    def apply(self, sim):
        pass

    def finalize(self, sim):
        fn = f'sim_{sim.scen.count:04d}_of_{sim.scen.total:04d}.tmp'
        sc.savetext(fn, string=str(sim))
        return


class reset_rng(cv.Intervention):
    """ Reset the random number seed on each day """
    def apply(self, sim):
        seed = sim['rand_seed'] + sim.t
        try:
            sim.set_seed(seed=seed)
        except:
            print(f'Could not set seed {seed}')
        return 


# Define the column names
column_info = {
    'es_prev_thresh': 'Minimum prevalence threshold for ES detection',
    'es_sensitivity': 'Sensitivity of ES to detection once above threshold',
    'es_specificity': 'Specificity of ES detection',
    'es_frequency': 'How frequently samples are taken',
    'es_n_samples': 'How many samples are taken each time',
    'es_trigger': 'The trigger rate for ES detections (of a new variant)',
    'clin_test_prob': 'Per-day probability of a person with symptoms testing',
    'clin_trigger': 'The trigger rate for clinical detections',
    'intervention_eff': 'The efficacy of the masking/distancing intervention',
    'variant_trans': 'Relative transmissibility of the new variant compared to wild type',
    'variant_sev': 'Severity of the new variant compared to wild type',
    'baseline_imm': 'Fraction of the population that has prior immunity to COVID',

    'label': 'Simulation label',
    'i': 'Simulation index (random seed)',
    'use_es': 'Whether ES is in use',
    'cum_infections': 'Cumulative infections',
    'cum_severe': 'Cumulative severe cases (hospitalizations)',
    'cum_critical': 'Cumulative critical cases (ICU)',
    'cum_deaths': 'Cumulative deaths',
    'cum_tests': 'Cumulative clinical tests',
    'cum_diagnoses': 'Cumulative diagnoses',

    'es_detections': 'Number of positive ES samples',
    'es_negatives': 'Number of negative ES samples',

    'yld': 'Cumulative years lived with disability',
    'yll': 'Cumulative years of life lost',
    'dalys': 'Cumulative DALYs',
}

# Define the default random scenario
default_rand_scen = sc.objdict(

    # ES parameters
    es_prev_thresh  = [0, 0.02], # Minimum prevalence threshold for ES detection
    es_sensitivity  = [0.25, 1.0], # Sensitivity of ES to detection once above threshold
    es_specificity  = [1.0, 1.0], # Specificity of ES detection
    es_frequency    = [1, 14, 'int'], # How frequently samples are taken
    es_n_samples    = [1, 5, 'int'], # How many samples are taken each time
    es_trigger      = [1, 1, 'int'], # The trigger rate for ES detections (of a new variant)

    # Intervention parameters
    clin_test_prob   = [0.001, 0.010], # Per-day probability of a person with symptoms testing
    clin_trigger     = [10, 30, 'int'], # The trigger rate for clinical detections
    intervention_eff = [0.1, 0.3], # The efficacy of the lockdown/mask/distancing intervention

    # Variant/immunity parameters
    variant_trans   = [2.0, 4.0], # Relative transmissibility of the new variant compared to wild type
    variant_sev     = [1.0, 3.0], # Severity of the new variant compared to wild type
    baseline_imm    = [0.10, 0.20], # Fraction of the population that has prior immunity to COVID
)

# Define the default grid scenario
default_grid_scen = sc.objdict(

    # ES parameters
    es_prev_thresh  = [0.001, 0.02], # Minimum prevalence threshold for ES detection
    es_sensitivity  = [0.25, 1.0], # Sensitivity of ES to detection once above threshold
    es_specificity  = [1.0, 1.0], # Specificity of ES detection
    es_frequency    = [1, 14], # How frequently samples are taken
    es_n_samples    = [1, 5], # How many samples are taken each time
    es_trigger      = [1], # The trigger rate for ES detections (of a new variant)

    # Intervention parameters
    clin_test_prob   = [0.001, 0.010], # Per-day probability of a person with symptoms testing
    clin_trigger     = [20], # The trigger rate for clinical detections
    intervention_eff = [0.30], # The efficacy of the lockdown/mask/distancing intervention

    # Variant/immunity parameters
    variant_trans   = [3.0], # Relative transmissibility of the new variant compared to wild type
    variant_sev     = [2.0], # Severity of the new variant compared to wild type
    baseline_imm    = [0.15], # Fraction of the population that has prior immunity to COVID
)

# Define the default line scenario
default_line_scen = sc.objdict(

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


# Default simulation parameters
default_pars = sc.objdict(
    pop_size     = 10_000, # Population size
    pop_infected = 0,
    start_day    = '2022-06-01', # First day of simulation
    end_day      = '2022-10-01', # Last day of simulation
    pop_type     = 'hybrid', # TODO: use SynthPops
    location     = None, # Location
    verbose      = -1, # How much detail to print (0 = none; -1 = one line per sim)
)

# Non-standard simulation parameters
other_pars = sc.objdict(
    days_prior   = 90, # Number of days prior to simulation to seed infections
    base_variant = 'omicron', # Base parameters for the new variant
    import_day   = 0, # Which day of the simulation to import the new variant
    n_imports    = 0, # Number of seed infections to introduce
)

# Which results to pull out for the summary
result_keys = [
    'cum_infections',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'cum_tests',
    'cum_diagnoses',
]


def get_path(sim, ext='sim'):
    """ Get a filepath from a simulation """
    return f'{sim.scen.filestem}.{ext}'


def run_sim(sim, load=True):
    """ Run a single sim and save to disk, or load if already run """
    path = get_path(sim, 'sim')
    if load and os.path.exists(path):
        sim = sc.load(path)
    else:
        try:
            sim.run()
            sim.shrink()
            sim.save(path)
        except Exception as E:
            print(f'WARNING, exception encountered! {E}')
            err = str(sim.scen)
            err += '\n\n'
            err += str(sc.traceback())
            sc.savetext(get_path(sim, 'err'), string=err)
    return sim


class Experiment(sc.prettyobj):
    """
    Define and run an ES experiment.
    """

    def __init__(self, label=None, n=1, method='line', scenario=None, sim_pars=None, 
                 rel_asymp_prob=0.02, unique_rand=False, load=True, verbose=True):
        self.label = label if label else 'default'
        self.n = n
        self.method = method
        if method == 'grid':
            self.scen = sc.mergedicts(default_grid_scen, scenario)
        elif method == 'rand':
            self.scen = sc.mergedicts(default_rand_scen, scenario)
        elif method == 'line':
            self.scen = sc.mergedicts(default_line_scen, scenario)
        self.sim_pars = sc.mergedicts(default_pars, sim_pars)
        self.rel_asymp_prob = rel_asymp_prob # How much less likely someone without symptoms is likely to test
        self.unique_rand = unique_rand
        self.load    = load
        self.verbose = verbose
        self.sims    = None
        self.msim    = None
        self.results = None
        return


    def rand_sample(self):
        """ Draw a uniform sample for the values in the scenario """
        pars = sc.objdict()
        for k,lims in self.scen.items():
            low = lims[0]
            high = lims[1]
            default = default_rand_scen[k]
            if len(default) == 3 and default[2] == 'int':
                val = np.random.randint(low, high+1)
            else:
                val = np.random.uniform(low, high)
            pars[k] = val
        return pars


    def create_sim(self, scen):
        """ Create a single sim """

        # Create interventions
        ints = [] # [reset_rng()] # Optionally reset the sim's random number seed
        if scen.baseline_imm:
            hist = cv.historical_wave(days_prior=other_pars.days_prior, prob=scen.baseline_imm) # Historical infections/vaccination
            ints += [hist]
        bt = beta_threshold(change=1-scen.intervention_eff, clin_trigger=scen.clin_trigger, es_trigger=scen.es_trigger, verbose=self.verbose)
        clin = cv.test_prob(symp_prob=scen.clin_test_prob, asymp_prob=scen.clin_test_prob*self.rel_asymp_prob) # Clinical testing
        es = ES( # Environmental surveillance
            prev_thresh = scen.es_prev_thresh,
            sensitivity = scen.es_sensitivity,
            specificity = scen.es_specificity,
            frequency   = scen.es_frequency,
            lag         = scen.es_lag,
            n_samples   = [0, scen.es_n_samples][scen.use_es], # Disable ES by using 0 samples
            verbose     = self.verbose,
        )
        ints += [clin, es, bt]
        ans = [dalys(), progress()]

        # Define variant
        if other_pars.n_imports:
            var = cv.variant(other_pars.base_variant, days=other_pars.import_day, n_imports=other_pars.n_imports)
            var.p['rel_beta'] = scen.variant_trans
            var.p['rel_severe_prob'] = scen.variant_sev
        else:
            var = None

        # Handle how random numbers are generated -- either unique for each sim, or only unique between replicates
        rand_seed = scen.count if self.unique_rand else scen.i
        if 'rand_seed' in self.sim_pars:
            rand_seed += self.sim_pars['rand_seed']

        # Create the sim
        sim = cv.Sim(self.sim_pars,
                     rand_seed=rand_seed,
                     label=scen.label,
                     interventions=ints,
                     analyzers=ans,
                     variants=var)
        sim.scen = sc.dcp(scen)

        return sim


    def create_grid_sims(self):
        self.sims = list()
        base_scen = sc.mergedicts(default_grid_scen, self.scen)
        scenkeys = base_scen.keys()
        scenvals = base_scen.values()
        all_scens = list(it.product(*scenvals))
        count = 0
        total = len(all_scens)*int(self.n)*2
        scens = []
        for entry in all_scens:
            for i in range(int(self.n)):
                for use_es in [0, 1]:
                    count += 1
                    scen = sc.objdict(dict(zip(scenkeys, entry)))
                    scen.i = i
                    scen.count = count
                    scen.total = total
                    scen.use_es = use_es
                    scen.label = f'Sim {count}/{total}, ES={use_es}'
                    scens.append(scen)
        
        self.sims = sc.parallelize(self.create_sim, scens)
        return


    def create_rand_sims(self):
        self.sims = list()
        count = 0
        total = int(self.n)*2
        for i in range(int(self.n)):
            for use_es in [0, 1]:
                count += 1
                scen = sc.objdict()
                scen.i = i
                scen.count = count
                scen.total = total
                scen.use_es = use_es
                scen.label = f'Sim {count}/{total}, ES={use_es}'
                scen = sc.mergedicts(scen, self.rand_sample())
                sim = self.create_sim(scen)
                self.sims.append(sim)
        return


    def default_scen(self):
        scen = sc.objdict({k:v[0] for k,v in self.scen.items()}) # Pull out default scenario
        return scen
    
    
    def create_line_sims(self):
        self.sims = list()
        default_scen = self.default_scen()
        all_scens = [default_scen]
        for key,val_lists in self.scen.items():
            if len(val_lists) == 2:
                for val in val_lists[1]:
                    this_scen = sc.dcp(default_scen)
                    this_scen[key] = val
                    all_scens.append(this_scen)
            
        count = 0
        total = len(all_scens)*int(self.n)*2
        scens = []
        self.filestems = sc.odict()
        for entry in all_scens:
            for i in range(int(self.n)):
                for use_es in [0, 1]:
                    count += 1
                    scen = sc.objdict(entry)
                    scen.i = i
                    scen.count = count
                    scen.total = total
                    scen.use_es = use_es
                    scen.label = f'Sim {count}/{total}, ES={use_es}'
                    scen.filestem = f'output_{self.label}/sim_{scen.count:05d}_of_{scen.total:05d}'
                    scens.append(scen)
                    self.filestems[scen.filestem] = False
        
        self.sims = sc.parallelize(self.create_sim, scens)
        return


    def create_sims(self):
        """ Create all simulations """
        if self.method == 'grid':
            self.create_grid_sims()
        elif self.method == 'rand':
            self.create_rand_sims()
        elif self.method == 'line':
            self.create_line_sims()
        return self
    
    
    def run_sims(self, die=False, **kwargs):
        """ Run all sims in parallel """
        sims_done = sc.parallelize(run_sim, self.sims, die=die, load=self.load, **kwargs)
        self.sims_done = sims_done
        return


    def run(self, n=None, **kwargs):
        """ Run the simulations in parallel """
        sc.heading('Running...')
        self.timer = sc.timer()
        if n is not None:
            self.n = n
        if not self.sims:
            self.create_sims()

        # Run and create results
        self.run_sims(**kwargs)
        self.compute_results()
        self.timer.toc()
        return self


    def compute_results(self):
        """ Parse the sim results into a dataframe """
        res = sc.autolist()
        for s in self.sims_done:
            entry = sc.dcp(s.scen)
            for key in result_keys:
                entry[key] = s.summary[key]
            es = s.get_intervention(ES)
            bt = s.get_intervention(beta_threshold)
            da = s.get_analyzer(dalys)
            entry.es_detections = sum(es.detections)
            entry.es_negatives = sum(es.negatives)
            entry.beta_start = bt.days[0] if len(bt.days) else np.nan
            entry.beta_days = bt.days_active
            entry.yld = da.yld
            entry.yll = da.yll
            entry.dalys = da.dalys
            path = get_path(s, 'json')
            sc.savejson(path, entry)
            res += entry
        self.results = sc.dataframe(res)
        return


    def export(self, filename=None, write_info=True, sheetname='Column info'):
        """ Export results to XLSX """
        if filename is None:
            filename = self.label + '.xlsx'
        self.results.to_excel(filename)
        if write_info:
            colinfo = [['Column name', 'Description']]
            for k,v in column_info.items():
                colinfo.append([k,v])
            S = sc.Spreadsheet(filename)
            S.openpyxl()
            S.wb.create_sheet(sheetname)
            S.writecells(vals=colinfo, sheetname=sheetname)
            S.save(filename)
        return