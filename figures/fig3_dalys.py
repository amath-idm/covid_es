"""
Plot DALYs and DALYs averted
"""

import sciris as sc
import pylab as pl
import utils as ut

do_save = True

#%% Load data

T = sc.timer()

dfs = ut.load_data(default=True, which='df')

    
#%% Plotting

cols = sc.objdict(disability='cornflowerblue', death='silver')

fig = ut.make_fig(height=6)

base = sc.dictobj()
diff = sc.dictobj()
base_es1 = sc.objdict()
absdiff = sc.objdict()

for l,loc in enumerate(ut.locations):
    
    df = dfs[loc]
    es0 = df[df.use_es==0]
    es1 = df[df.use_es==1]
    base[loc] = es0.mean()
    base_es1[loc] = es1.mean()
    absdiff[loc] = es0.mean() - es1.mean()
    diff[loc] = (es0.mean() - es1.mean())/es0.mean()*100

res = sc.objdict()
res.base = sc.cat([
    [base.malawi.yld, base.nepal.yld],
    [base.malawi.yll, base.nepal.yll],
])
res.diff = sc.cat([
    [diff.malawi.yld, diff.nepal.yld],
    [diff.malawi.yll, diff.nepal.yll],
])
totdiff = [absdiff.malawi.dalys, absdiff.nepal.dalys]
base_es = [base_es1.malawi.dalys, base_es1.nepal.dalys]
    
for i,val in res.enumvals():
    ax = pl.subplot(1,2,i+1)
    pl.title(['Total DALYs', 'DALYs averted by ES (%)'][i], fontweight='bold')
    
    sc.stackedbar(val, labels=['YLD', 'YLL'], colors=cols.values())
    if i == 0:
        pl.bar([0, 1], totdiff, bottom=base_es, edgecolor='k', hatch='///', lw=0.5, fc='none', label='DALYs averted by ES')
        sc.commaticks()
    else:
        ax.set_yticklabels([f'{int(x)}%' for x in ax.get_yticks()]) # Raises a warning, but correct result
    
    pl.xticks([0, 1], ut.labels)
    pl.legend()
    sc.orderlegend(reverse=True, frameon=False)
    sc.boxoff()

# Tidy
ut.tidy('fig3_dalys.png', do_save=do_save)
T.toc()