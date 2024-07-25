"""
Look at the relationship between beta_days and dalys_averted.
"""

import sciris as sc
import pylab as pl
import scipy.stats as sps
T = sc.timer()

do_filter = 1
do_save = 0
fn = 'nepal_oct23_esd_raw.df'


data = sc.load(fn)
raw = data['esd']

if do_filter:
    max_days = 100
    raw = raw[raw.beta_days<max_days]
    
effs = raw.intervention_eff.unique()

sc.options(dpi=150)
pl.figure(figsize=(20,16))

res = dict()

for i,eff in enumerate(effs):

    # Calculate mean
    esd = raw[raw.intervention_eff==eff]
    g = esd.groupby('beta_days')
    m = g.mean()
    
    # Do regressions
    f = sc.objdict()
    f.all  = sps.linregress(esd.beta_days, esd.dalys_averted)
    f.mean = sps.linregress(m.index, m.dalys_averted)
    
    pl.subplot(3,2,2*i+1)
    pl.scatter(esd.beta_days, esd.dalys_averted, alpha=0.1)
    pl.plot(esd.beta_days, f.all.slope*esd.beta_days+f.all.intercept, lw=2, c='k')
    pl.title(f'Slope: {f.all.slope:0.2f} DALYs/day')
    
    pl.subplot(3,2,2*i+2)
    pl.scatter(m.index, m.dalys_averted, alpha=0.8)
    pl.plot(esd.beta_days, f.mean.slope*esd.beta_days+f.mean.intercept, lw=2, c='k')
    pl.title(f'Slope: {f.mean.slope:0.2f} DALYs/day')
    
    res[eff] = f.all.slope

if do_save:
    sc.savejson('nepal_slopes_oct23.json', res)
pl.show()
T.toc()

