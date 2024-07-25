"""
Plot sensitivity analysis
"""

import numpy as np
import sciris as sc
import pylab as pl
import utils as ut

do_save = True

#%% Load and wrangle data

sc.heading('Loading and analyzing data...')

T = sc.timer()

es0s = ut.load_data(which='es0')
esds = ut.load_data(which='esd')
default_esds = ut.load_data(which='esd', default=True)

labels = sc.objdict({
    'es_prev_thresh': 'Population COVID prevalence\nthreshold for ES detection (%)',
    'es_sensitivity': 'Per-sample sensitivity of ES to detection (%)',
    'es_frequency': 'Time between ES sample collections (days)',
    'es_n_samples': 'Number of ES samples collected',
    'es_lag': 'Time from ES collection to response (days)',
    'clin_trigger': 'Clinical threshold for triggering\nintervention (# positives)',
    'clin_test_prob': 'Probability of clinical diagnosis\nof a COVID infection (%)',
    'intervention_eff': 'Efficacy of intervention\n(distancing, mask mandate, etc.) (%)',
})
labels = sc.objdict({
    'es_prev_thresh':  'ES prevalence\nthreshold (%)',
    'es_sensitivity':  'ES sensitivity (%)',
    'es_frequency':    'ES collection\nfrequency (days)',
    'es_n_samples':    'Number of\nES samples',
    'es_lag':          'ES processing\ntime (days)',
    'clin_trigger':    'Clinical trigger\n(# positives)',
    'clin_test_prob':  'Clinical test\nprobability (%)',
    'intervention_eff':'Intervention\nefficacy (%)',
})

columns = labels.keys()

# Find unique values
defaults  = sc.objdict()
sensdict = sc.objdict()
for loc in ut.locations: # Should be the same, but just in case
    defaults[loc] = ut.default_filter[loc]
    sensdict[loc] = sc.objdict()
    for col in columns:
        unique = esds[loc][col].unique()
        unique = [v for v in unique if v != defaults[loc][col]]
        sensdict[loc][col] = unique
        esds[loc][col] = es0s[loc][col]

def get_stats(series, z=1.96):
    mean = int(round(series.mean()))
    sem = int(round(series.sem()*z))
    out = sc.objdict(best=mean, low=mean-sem, high=mean+sem)
    return out

def fmt(num, pct=False):
    string = f'{num:n}'
    if pct:
        string += '%'
    return string


# Handle defaults
dres = sc.objdict()
for loc in ut.locations:
    df = default_esds[loc]
    dres[loc] = get_stats(df.dalys_averted)

# Do analysis
count = sc.objdict()
sres = sc.objdict()
for l,loc in enumerate(ut.locations):
    print(f'Processing {loc}...')
    count[loc] = 1 # For default values and and space
    sres[loc] = sc.objdict()
    for col in columns:
        print(f'  Processing {col}...')
        count[loc] += 1
        df = esds[loc]
        sres[loc][col] = sc.objdict()
        for val in sensdict[loc][col]:
            count[loc] += 1
            averted = df[df[col]==val].dalys_averted
            if col == 'clin_test_prob':
                vkey = fmt(val*10*100, pct=True) # To handle multiple day testing
            elif '%' in labels[col]:
                vkey = fmt(val*100, pct=True)
            else:
                vkey = fmt(val)
            sres[loc][col][vkey] = get_stats(averted)

    
#%% Plotting

sc.heading('Plotting...')

fig = ut.make_fig(height=8)

origcount = count.malawi
yparlabs = dict()
yticklabs = dict()

dy1 = 0.5
dy2 = 1.0
height = dy2*0.8
axs = sc.objdict()
for l,loc in enumerate(ut.locations):
    axs[loc] = pl.subplot(1,2,l+1)
    pl.title(ut.labeldict[loc], fontweight='bold')
    pl.xlabel('DALYs averted')
    sc.commaticks(axis='x')
    pl.axvline(0, lw=0.5, c='k')
    sc.boxoff()
    
    yparlabs[count[loc]] = 'Default'
    vval = dres[loc]
    pl.barh(count[loc], vval.best, height=height, facecolor='steelblue')
    pl.plot([vval.low, vval.high], [count[loc]]*2, c='k')
    count[loc] -= 2*dy1
    
    for col,cval in sres[loc].items():
        count[loc] -= dy1
        for vkey,vval in cval.items():
            yparlabs[count[loc]] = f'{labels[col]}'
            yticklabs[count[loc]] = vkey
            pl.barh(count[loc], vval.best, height=height, facecolor='deepskyblue')
            pl.plot([vval.low, vval.high], [count[loc]]*2, c='k')
            count[loc] -= dy2
    
    pl.yticks(list(yticklabs.keys()), list(yticklabs.values()))
    pl.grid(True, axis='x', lw=1, alpha=0.5)
    pl.ylim([count[loc]+0.3, origcount+height])
            
# Plot labels
ypdict = sc.ddict(list)
for k,v in yparlabs.items():
    ypdict[v].append(k)
for k,v in ypdict.items():
    ypdict[k] = np.mean(v)

for k,v in ypdict.items():
    axs.malawi.text(x=-1500, y=v, s=k, fontweight='bold', verticalalignment='center')


# Stats
ie = sc.objdict()
for loc in ut.locations:
    ie[loc] = dict()
    for eff in ['10%', '30%']:
        ie[loc][eff] = sres[loc].intervention_eff[eff].best
    ie[loc]['factor'] = round(ie[loc]['30%']/ie[loc]['10%'])
string = f'''
Intervention efficacy also had a relatively large and nonlinear impact on
estimated DALYs averted. Increasing the intervention efficacy from 10% to 30%
increased the number of DALYs averted from {ie.malawi["10%"]} to {ie.malawi["30%"]} in Malawi
and {ie.nepal["10%"]} to {ie.nepal["30%"]} in Nepal, increases by factors of
{ie.malawi["factor"]} and {ie.nepal["factor"]} respectively.
'''
string = string.replace('\n',' ')
print(string)


# Tidy
ut.tidy('fig5_sensitivity.png', do_save=do_save, left=0.18)
T.toc()