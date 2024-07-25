"""
Shared settings for figures
"""

import sciris as sc
import pylab as pl
import covasim_es as ces
from malawi_scripts import sweep_lines as msl
from nepal_scripts  import sweep_lines as nsl

locations = ['malawi', 'nepal']
labels = ['Blantyre', 'Kathmandu']
labeldict = sc.objdict({k:v for k,v in zip(locations, labels)})

# Define things used globally
modules = dict(malawi=msl, nepal=nsl)
default_scen = sc.objdict()
default_filter = sc.objdict()
for loc,module in modules.items():
    default_scen[loc] = module.make_scenario(default=True)
    default_filter[loc] = sc.objdict({k:v[0] for k,v in default_scen[loc].items()})

#%% Run/load helper functions

def run_exps(n=10, load=True, verbose=True, **kwargs):
    
    sc.heading('Creating experiments (NB, slow the first time)...')
    exp = sc.objdict()
    for loc,module in modules.items():
        scenario = sc.mergedicts(default_scen[loc], **kwargs)
        pars = sc.dcp(module.sim_pars)
        e = ces.Experiment(label=loc, n=n, scenario=scenario, method='line', sim_pars=pars, load=load, verbose=verbose)    
        exp[loc] = e

    for e in exp.values():
        e.run()
        
    return exp


def load_data(filters=None, do_filter=None, which='df', default=False, verbose=False):
    data = sc.objdict()
    for loc in locations:
        if default:
            filters = sc.dcp(default_filter[loc])
        filters = sc.mergedicts(filters)
        path = sc.thispath() / f'../results/{loc}_esd_raw.df'
        raw = sc.load(path)
        df = raw[which]
        for key,val in filters.items():
            l1 = len(df)
            df = df[df[key]==val]
            l2 = len(df)
            if verbose: print(key, val, l1, l2)
        if do_filter and which=='esd': # Used to exclude certain outliers
            max_days = 100
            df = df[df.beta_days<max_days]
        data[loc] = df
    return data
    

#%% Plotting helper functions

def make_fig(width=10, height=8, **kwargs):
    defaults = dict(dpi=200, font='Rosario', fontsize=12)
    kw = sc.mergedicts(defaults, kwargs)
    sc.options(**kw)
    fig = pl.figure(figsize=(width, height))
    return fig


def fix_alpha(leg):
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    return


def tidy(filename=None, do_save=False, **kwargs):
    sc.figlayout(**kwargs)
    pl.show()
    if do_save:
        sc.savefig(filename)
    return