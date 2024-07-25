"""
Calculate costs and plot graphs
"""

import os
import pylab as pl
import numpy as np
import sciris as sc
import covasim as cv
import sweep_lines as sl

filestem  = 'nepal_oct23'
subfolder = 'results_oct23'
country   = 'nepal'
esd_file  = f'{filestem}_esd.df'
excel_file = f'{filestem}.xlsx'
df_file = f'{filestem}.df'
json_file = 'nepal_slopes_oct23.json'

do_calc = 1
do_plot = 1
do_save = 1
use_ave = 0
use_sem = 0
nonparametric = False # Swap between mean and median
plot_scatter = False

# DALYs averted per day of lockdown, from compute_slope.py
use_statistical_dalys = 1

T = sc.timer()

# Get scenario
scen = sl.make_scenario()

# Define sampling function
def unif(low, high):
    return np.random.uniform(low, high, size=len(df))

# Costs
ex = 118.31 # USD to NPR exchange rate as of 2021
c = sc.objdict()
n_months = 6
year_factor = 12/n_months
per_capita_gdp = 1155.14
covid_prod_loss_mild = 0.10 # Estimate
covid_prod_loss_severe = 0.50 # Estimate
intervention_prod_loss = 0.1 # Estimate
pop_size = 2.734802e6
mild_days = 5#unif(3,7)
severe_days = 21.5#unif(15, 28)
age_dist = cv.data.get_age_distribution(country)
working_frac = age_dist[sc.findinds((age_dist[:,0] >= 20) * (age_dist[:,0] < 60)),-1].sum()
working_pop = pop_size*working_frac
c.severe = 25_000/ex#unif(10_000, 40_000)/ex # From https://github.com/amath-idm/covasim_es/issues/35
c.mild = (215+2140)/2/ex#unif(215, 2140)/ex # From https://github.com/amath-idm/covasim_es/issues/35
c.es_base = 8470.46 # From Mercy email 2022-05-07 + cost calculation script
c.es_per_sample = 86.31
c.test = 1500/ex
days_to_gdp = working_frac*per_capita_gdp/365 # Convert a number of person-days of lost productivity to a dollar estimate
willingness_threshold = per_capita_gdp*3

max_icer = 100e3
min_icer = 0

if do_calc:
    sc.heading('Loading data...')
    
    if do_save and not os.path.exists(subfolder):
        os.mkdir(subfolder)
    
    # Load data
    try:
        df = sc.load(df_file)
    except:
        print(f'Binary dataframe {df_file} not found, loading Excel')
        df = sc.dataframe.read_excel(excel_file)
        sc.save(df_file, df)
        
    for key in ['label', 'filestem']:
        del df[key] # Breaks subtraction later without this
    
    # Perform optional filtering
    do_filter = False # Used for grid sweeps to zoom into a better area
    if do_filter:
        df = df[df.intervention_eff == df.intervention_eff.max()]
        df = df[df.es_sensitivity == df.es_sensitivity.max()]
        df = df[df.es_frequency == df.es_frequency.min()]
    
    # Calculate new columns
    df['cum_mild'] = df.cum_infections - df.cum_severe
    df['cost_es'] = df.use_es * (c.es_base*n_months + c.es_per_sample*(df.es_detections+df.es_negatives))
    df['cost_mild'] = df.cum_mild*c.mild
    df['cost_severe'] = df.cum_severe * c.severe
    df['cost_covid'] = df.cost_mild + df.cost_severe
    df['cost_test'] = df.cum_tests * c.test
    df['cost_covid_productivity'] = (df.cum_mild*mild_days*covid_prod_loss_mild + df.cum_severe*severe_days*covid_prod_loss_severe)*days_to_gdp
    df['cost_intervention'] = df.beta_days*df.intervention_eff*intervention_prod_loss*days_to_gdp*pop_size
    df['cost'] = df.cost_es + df.cost_covid + df.cost_test
    df['full_cost'] = df.cost + df.cost_covid_productivity + df.cost_intervention
    for col in df.columns:
        if 'cost' in col or 'cum' in col:
            print(f'Note: column {col} is being corrected by duration-to-year {year_factor}')
            df[col] *= year_factor
    
    # Calculate differences
    es0 = df[df.use_es==0].reset_index()
    es1 = df[df.use_es==1].reset_index()
    esd = es1 - es0
    par_cols = []
    for col in esd.columns:
        if esd[col].sum() == 0:
            if col not in ['i', 'total']:
                par_cols.append(col)
            esd[col] = es0[col]
            
    # Compute average before computing additional statistics
    if use_ave:
        g0 = es0.groupby(par_cols).mean().reset_index()
        g1 = es1.groupby(par_cols).mean().reset_index()
        esd = g1 - g0
        for col in par_cols:
            esd[col] = g0[col]
    
    # Compute derived quantities
    if use_statistical_dalys:
        slopes = sc.loadjson(json_file)
        esd['dalys_per_day'] = np.nan
        effs = esd.intervention_eff.unique()
        for eff in effs:
            esd.dalys_per_day[esd.intervention_eff==eff] = slopes[str(eff)]
        assert sum(np.isnan(esd.dalys_per_day)) == 0
        esd['dalys_averted'] = esd.beta_days * esd.dalys_per_day
    else:
        esd['dalys_averted'] = -esd.dalys
    esd['pos_dalys_averted'] = esd.dalys_averted > 0
    esd['pos_cost'] = esd.cost > 0
    esd['pos_full_cost'] = esd.full_cost > 0
    esd['cost_saving'] = esd.cost < 0
    esd['full_cost_saving'] = esd.full_cost < 0
    esd['cost_effective'] = esd.cost < willingness_threshold
    esd['full_cost_effective'] = esd.full_cost < willingness_threshold
    esd['valid_icer'] = esd.pos_dalys_averted * esd.pos_cost
    esd['valid_full_icer'] = esd.pos_dalys_averted * esd.pos_full_cost
    esd['icer'] = esd.cost / esd.dalys_averted
    esd['full_icer'] = esd.full_cost / esd.dalys_averted
    
    # Do more data cleaning
    neg_averted = ~esd.pos_dalys_averted
    cap_icers = False
    if cap_icers:
        esd.icer[neg_averted] = max_icer
        esd.icer[esd.icer>max_icer] = max_icer
        esd.icer[esd.icer<min_icer] = min_icer
        
        esd.full_icer[neg_averted] = max_icer
        esd.full_icer[esd.full_icer>max_icer] = max_icer
        esd.full_icer[esd.full_icer<min_icer] = min_icer
    
    esd = esd.replace([np.inf, -np.inf], np.nan).fillna(0) # Remove infinities
    
    # NB, save regardless of do_save value
    sc.save(esd_file, dict(es0=es0, es1=es1, esd=esd, par_cols=par_cols, df=df))

else:
    es0, es1, esd, par_cols, df = sc.load(esd_file).values()


# Handle stats
stats = sc.dataframe(columns=[
    'parameter',
    'label',
    'value',
    'dalys_averted_best',
    'dalys_averted_low',
    'dalys_averted_high',
    'cost_best',
    'cost_low',
    'cost_high',
    'full_cost_best',
    'full_cost_low',
    'full_cost_high',
    'icer_best',
    'icer_low',
    'icer_high',
    'full_icer_best',
    'full_icer_low',
    'full_icer_high',
    'cost_saving',
    'full_cost_saving',
    'cost_effective',
    'full_cost_effective',
    ]
)

sumcols = [
    'parameter',
    'label',
    'value',
    'dalys_averted',
    'cost',
    'full_cost',
    'icer',
    'full_icer',
]

sumstats = sc.dataframe(columns=sumcols)

# Optionally plot
sc.heading('Plotting...')
sc.options(dpi=150)
labels = {
    'es_prev_thresh': 'Population COVID prevalence\nthreshold for ES detection (%)',
    'es_sensitivity': 'Per-sample sensitivity of ES to detection (%)',
    'es_frequency': 'Time between ES sample collections (days)',
    'es_n_samples': 'Number of ES samples collected',
    'es_trigger': 'ES threshold for triggering\nintervention (# detections)',
    'es_lag': 'Time from ES collection to response (days)',
    'clin_trigger': 'Clinical threshold for triggering\nintervention (# positives)',
    'clin_test_prob': 'Probability of clinical diagnosis\nof a COVID infection (%)',
    'intervention_eff': 'Efficacy of intervention\n(distancing, mask mandate, etc.) (%)',
}
final_cols = []
nlist = []
z = 1.96 # Convert from e.g. an SEM to a 95% confidence interval
# qlow = 0.025
# qhigh = 0.975
for col in par_cols:
    unique = esd[col].unique()
    n = len(unique)
    if n > 1:
        final_cols.append(col)
        nlist.append(n)

defaults = {col:scen[col][0] for col in final_cols}
all_defaults = [(row == list(defaults.values())).all() for _,row in esd[final_cols].iterrows()]

for i, (col,n) in enumerate(zip(final_cols, nlist)):
    
    x = np.arange(n)
        
    # Filter out non-relevant default values
    #esd[esd.valid_icer].groupby(col) # For removing invalid ICERs (deprecated)
    valid = (esd[col] != defaults[col]) | all_defaults # Pull out valid rows
    thisesd = esd[valid]
    
    # Do the groupbys
    g = thisesd.groupby(col)
    if nonparametric:
        best = g.median()
        if use_sem:
            low  = best - g.sem()*z
            high = best + g.sem()*z
        else:
            low  = best - g.std()*z
            high = best + g.std()*z
        # low  = g.quantile(qlow)
        # high = g.quantile(qhigh)
    else:
        best = g.mean()
        if use_sem:
            low  = best - g.sem()*z
            high = best + g.sem()*z
        else:
            low  = best - g.std()*z
            high = best + g.std()*z
    std = g.std()
    vals = best.dalys_averted
    sems = g.sem()
    yerr = np.array([best.dalys_averted - low.dalys_averted, high.dalys_averted - best.dalys_averted])
    
    xlabels = g.mean().index
    
    dind = sc.findinds(xlabels, defaults[col])
    label = labels[col]
    if col in ['es_prev_thresh', 'clin_test_prob', 'intervention_eff', 'es_sensitivity']:
        xlabels *= 100
    if col in ['clin_test_prob']: # Infectious for ~10 days
        xlabels *= 10
    xticks = [f'{v:n}' for v in xlabels]
    
    # Assemble stats
    n_entries = len(xlabels)
    rows = sc.objdict(parameter=[col]*n_entries, label=[label.replace('\n',' ')]*n_entries, value=xlabels)
    sumrows = sc.dcp(rows)
    
    core = ['dalys_averted', 'cost', 'full_cost', 'icer', 'full_icer']
    extras = ['cost_es', 'cost_covid', 'cost_test', 'cost_covid_productivity', 'cost_intervention']
    for key in core + extras:
        rows[f'{key}_best'] = best[key]
        rows[f'{key}_low']  = low[key]
        rows[f'{key}_high'] = high[key]
        
    cea = ['cost_saving', 'full_cost_saving', 'cost_effective', 'full_cost_effective']
    for key in cea:
        rows[key] = best[key]
    
    two_round = ['cost_saving', 'cost_effective', 'full_cost_saving', 'full_cost_effective']
    no_round = ['parameter', 'label', 'value']
    for k in rows.keys():
        if k in two_round:
            rows[k] = np.round(rows[k], decimals=2)
        elif k not in no_round:
            rows[k] = np.round(rows[k])
    
    new = sc.dataframe(rows)
    stats.appendrow(new)
    
    # Handle summary
    for key in sumcols:
        if key not in sumrows:
            sumrows[key] = best[key]
            if key not in no_round:
                sumrows[key] = np.round(sumrows[key])
    sumnew = sc.dataframe(sumrows)
    sumstats.appendrow(sumnew)
        
    # Plotting
    if do_plot:
        pl.figure(figsize=(5,3))
        pl.axhline()
        pl.bar(x, vals, facecolor='deepskyblue')
        pl.bar(x[dind], vals.values[dind], facecolor='steelblue', label='Default')
        pl.errorbar(x, vals, yerr=yerr, ecolor='k', fmt='none')
        pl.xlabel(label)
        pl.xticks(x, xticks)
        pl.ylabel('DALYs averted by ES')
        sc.commaticks()
        sc.boxoff()
        sc.figlayout()
        if do_save:
            sc.savefig(f'{subfolder}/{filestem}_{col}.png', verbose=True)
                
    pl.show()

# Optionally save
stem = f'{subfolder}/{filestem}_'
if do_save:
    sc.heading('Saving...')
    esd.to_excel(f'{stem}cea.xlsx')
    stats.to_excel(f'{stem}stats.xlsx')
    sumstats.to_excel(f'{stem}summary.xlsx')
    # es0.to_excel(f'{stem}no_es.xlsx')
    # es1.to_excel(f'{stem}with_es.xlsx')
    # df.to_excel(f'{stem}raw.xlsx')
    

#%% Print out results
sc.heading('Results:')

def qstr(vec):
    q = [0.50, 0.25, 0.75]
    vals = vec.quantile(q).values
    string = f'{vals[0]:0,.0f} [IQR: {vals[1]:0,.0f} – {vals[2]:0,.0f}]'
    return string

for k,label in dict(dalys='DALYs', cost='Costs', full_cost='Full costs').items():
    print(f"""
{label}
      No ES: {qstr(es0[k])}
    With ES: {qstr(es1[k])}
   {label} averted: {qstr(-esd[k])}
""")

if do_plot and plot_scatter:
    sc.heading('Plotting scatter...')
    f = 1e6
    pl.figure()
    pl.axhline()
    eflist = esd.es_frequency.unique()
    nslist = esd.es_n_samples.unique()
    ielist = esd.intervention_eff.unique()
    colors = sc.gridcolors(len(eflist)*len(nslist)*len(ielist))
    count = -1
    for e,ef in enumerate(eflist):
        for n,ns in enumerate(nslist):
            for i,ie in enumerate(ielist):
                count += 1
                label = f'ES freq.: {ef} day(s); ES samples: {ns}; intervention eff.: {ie*100}%'
                inds = (esd.es_frequency == ef) * (esd.intervention_eff == ie) * (esd.es_n_samples == ns)
                pl.scatter(es0.full_cost[inds]/f, esd.full_cost[inds]/f, alpha=0.5, c=[colors[count]], label=label)
    pl.legend()
    pl.xlabel('Full cost without ES ($US millions)')
    pl.ylabel('Cost difference with ES ($US millions)')
    sc.boxoff()
    if do_save:
        sc.savefig(f'{subfolder}/{country}_{filestem}_scatter.png')


sc.heading('Done.')
T.toc()
