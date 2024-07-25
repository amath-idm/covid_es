"""
Plot regression results between number of days of intervention and DALYs averted.
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import pylab as pl
import utils as ut

T = sc.timer()

do_filter = 1
do_save = 0

# Filter to the default scenario -- needs to match sweep_lines.py
filters = sc.objdict(
    intervention_eff = 0.2,
)

esds = ut.load_data(filters=filters, do_filter=do_filter, which='esd')


#%% Plotting
fig = ut.make_fig()
cols = sc.objdict(sim='lightseagreen', mean='darkgreen')

for l,loc in enumerate(ut.locations):
    
    esd = esds[loc] # ES difference

    # Calculate mean
    g = esd.groupby('beta_days')
    m = g.mean()
    
    # Do regressions
    f = sc.objdict()
    f.all  = sps.linregress(esd.beta_days, esd.dalys_averted)
    f.mean = sps.linregress(m.index, m.dalys_averted)
    
    # Plot
    pl.subplot(1,2,l+1)
    rnds = 0.2*(np.random.rand(len(esd))-0.5)
    pl.scatter(esd.beta_days+rnds, esd.dalys_averted, alpha=0.03, lw=0, c=cols.sim, label='Individual simulation')
    pl.scatter(m.index, m.dalys_averted, s=50, alpha=0.5, lw=0, c=cols.mean, label='Mean value per day', zorder=1)
    pl.plot(esd.beta_days, f.all.slope*esd.beta_days+f.all.intercept, lw=2, c='k', label='Line of best fit', zorder=0)
    pl.title(f'{ut.labeldict[loc]}: {f.all.slope:0.1f} DALYs/day', fontweight='bold')
    if not l: pl.ylabel('DALYs averted by ES')
    pl.xlabel('Number of additional days of intervention with ES')
    
    # Tidy
    if l:
        leg = pl.legend(loc='lower right', frameon=False)
        ut.fix_alpha(leg)
    pl.xlim(left=0)
    lim = dict(malawi=5e3, nepal=15e3)[loc]
    tickspace = dict(malawi=1e3, nepal=3e3)[loc]
    yticks = np.arange(-lim, lim+1, tickspace)
    pl.yticks(yticks)
    pl.ylim(bottom=-lim, top=lim)
    sc.boxoff()
    sc.commaticks()

    # Quantitative results
    q = esd.beta_days.quantile([0.5, 0.25, 0.75])
    print(f'{q[0.5]} (IQR: {q[0.25]}, {q[0.75]})')
    
# Tidy up
ut.tidy('fig2_regression.png', do_save=do_save)
T.toc()

