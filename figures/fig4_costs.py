"""
Plot costs
"""

import numpy as np
import sciris as sc
import pylab as pl
import utils as ut

do_save = True
factor = 1e6

#%% Load data

T = sc.timer()

dfs = ut.load_data(default=True, which='df')

labels = sc.objdict(
    cost_mild = 'Mild COVID-19',
    cost_severe = 'Severe COVID-19',
    cost_test = 'Clinical testing',
    cost_es = 'Environmental surveillance',
    cost_covid_productivity = 'COVID-19-related productivity loss',
    cost_intervention = 'Intervention-related productivity loss',
)
cols = sc.objdict(
    cost_mild = 'lightcoral',
    cost_severe = 'firebrick',
    cost_test = 'plum',
    cost_es = 'xkcd:baby poop green',
    cost_covid_productivity = 'skyblue',
    cost_intervention = 'steelblue',
)

res = sc.objdict()
data = sc.objdict()
for loc in ut.locations:
    df = dfs[loc]
    res[loc] = sc.objdict()
    arr = []
    for use_es in [0, 1]:
        edf = df[df.use_es==use_es]
        m = edf.mean()
        arr.append([m[col] for col in cols.keys()])
        res[loc][f'dalys_es{use_es}'] = m.dalys
        res[loc][f'costs_h{use_es}']  = m.cost
        res[loc][f'costs_f{use_es}']  = m.full_cost
        if use_es == 1:
            res[loc]['cost_test'] = m.cost_test
            res[loc]['cost_es'] = m.cost_es
            
    data[loc] = np.vstack(arr).T/factor
    res[loc]['dalys_esd'] = res[loc].dalys_es0 - res[loc].dalys_es1
    res[loc]['costs_hd']  = res[loc].costs_h1  - res[loc].costs_h0
    res[loc]['costs_fd']  = res[loc].costs_f1  - res[loc].costs_f0
    res[loc]['icer_h']    = res[loc].costs_hd/res[loc].dalys_esd
    res[loc]['icer_f']    = res[loc].costs_fd/res[loc].dalys_esd

    
#%% Plotting

fig = ut.make_fig(height=6)

for l,loc in enumerate(ut.locations):
    ax = pl.subplot(1,2,l+1)
    pl.title(ut.labels[l], fontweight='bold')
    
    sc.stackedbar(data[loc], labels=labels.values(), colors=cols.values())
    
    sc.commaticks()
    pl.xticks([0, 1], ['Without ES', 'With ES'])
    pl.ylabel('Cost (US$ millions)')
    if not l:
        pl.ylim(top=12)
        pl.legend()
        sc.orderlegend(reverse=True, frameon=False, loc='upper left')
    else:
        pl.ylim(top=120)
    sc.boxoff()


#%% Write results table

def fmt(key):
    def sig(res):
        return f'{int(res):n}'
    prefix = '$' if 'cost' in key else ''
    string = f'\t{prefix+sig(res.malawi[key])}\t{prefix+sig(res.nepal[key])}'
    return string

string = f'''
Total DALYs (without ES){fmt('dalys_es0')}
Total DALYs (with ES){fmt('dalys_es1')}
DALYs averted (with ES){fmt('dalys_esd')}
Clinical testing costs{fmt('cost_test')}
ES program costs{fmt('cost_es')}
Health system costs (without ES){fmt('costs_h0')}
Health system costs (with ES){fmt('costs_h1')}
Health system cost difference{fmt('costs_hd')}
Full costs (without ES){fmt('costs_f0')}
Full costs (with ES){fmt('costs_f1')}
Full cost difference{fmt('costs_fd')}
ICER (health system costs){fmt('icer_h')}
ICER (full costs){fmt('icer_f')}
'''
string = string.replace('$-', '-$').replace('-','–')
print(string)
    

# Tidy
ut.tidy('fig4_costs_v2.png', do_save=do_save)
T.toc()