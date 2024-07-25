"""
Plot example time series
"""

#%% Setup

import numpy as np
import sciris as sc
import pylab as pl
import utils as ut

n = 10
verbose = True
do_save = True
suppl = True

T = sc.timer()


#%% Create and run
kw = dict(clin_test_prob=[0.0]) if suppl else {}
exps = ut.run_exps(n=n, load=False, verbose=verbose, **kw)


#%% Plotting

sc.heading('Plotting...')
es_colors = ['plum', 'xkcd:baby poop green']
kw = dict(alpha=0.5, lw=2)

def tidy_ax(loc):
    pl.xticks(x, x)
    sc.boxoff()
    sc.setylim()
    sc.commaticks()
    pl.xlim(left=0)
    return
    
fig = ut.make_fig()
for l,loc in enumerate(ut.locations):
    es_names = {'Without ES':0, 'With ES':0}
    exp = exps[loc]
    results = [sc.mergedicts(sim.results, sim.scen) for sim in exp.sims_done]
    x = np.arange(0, 181, 30)
    
    pl.subplot(3,2,l+1)
    pl.title(ut.labeldict[loc], fontweight='bold')
    if not l: pl.ylabel('Daily new infections')
    for i,res in enumerate(results):
        label = None
        name = list(es_names.keys())[res.use_es]
        if not es_names[name]:
            label = name
            es_names[name] += 1
        pl.plot(res.new_infections, c=es_colors[res.use_es], label=label, **kw)
    tidy_ax(loc)
    if l:
        leg = pl.legend(frameon=False)
        ut.fix_alpha(leg)
    
    pl.subplot(3,2,l+3)
    if not l: pl.ylabel('Cumulative infections')
    for res in results:
        pl.plot(res.cum_infections, c=es_colors[res.use_es], **kw)
    tidy_ax(loc)
    
    pl.subplot(3,2,l+5)
    if not l: pl.ylabel('Cumulative deaths')
    for res in results:
        pl.plot(res.cum_deaths, c=es_colors[res.use_es], **kw)
    tidy_ax(loc)
    pl.xlabel('Days since variant introduction')
    
    
# Tidy up
filename = 'figS1_timeseries.png' if suppl else 'fig1_timeseries.png'
ut.tidy(filename, do_save=do_save)
T.toc('Script finished after')