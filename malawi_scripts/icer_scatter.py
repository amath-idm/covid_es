'''
Make a scatterplot of ICER values
'''

import numpy as np
import sciris as sc
import pylab as pl
import matplotlib as mpl

T = sc.timer()

do_save = True
filestem  = 'malawi_sep01'
subfolder = 'results_sep04'
esd_file = f'{filestem}_esd.df'
figstem = f'{subfolder}/{filestem}_'
data = sc.load(esd_file)
esd = data['esd']

max_icer = 2e3
min_icer = -max_icer
threshold = 61 # From Lucky email Aug. 22nd; was 411*3


sc.options(dpi=150)
fig = pl.figure(figsize=(16,10))
lkw = dict(c='k', lw=0.5)

def manualcolorbar(ax, cmap, vmin, vmax, vcenter, label):
    '''
    Make a manual colorbar
    '''
    import matplotlib as mpl
    norm = sc.midpointnorm(vcenter, vmin, vmax)
    cb = pl.colorbar()
    cb1 = mpl.colorbar.ColorbarBase(cb.ax, cmap=cmap, norm=norm)
    cb1.set_label(label)
    return


# Plot scatters
stride = 1
cmap1 = sc.orangebluecolormap()
cmap2 = mpl.cm.PiYG
icerlabel = 'ICER (US$/DALY averted)'
skw = dict(alpha=0.7, lw=0.1, edgecolor='k')
for i,(label,costvar,icervar) in enumerate([['Direct costs/ICERs', 'cost', 'icer'], ['Full costs/ICERs', 'full_cost', 'full_icer']]):
    
    da = esd['dalys_averted']
    cost = esd[costvar]
    icer = esd[icervar]
    icer = np.clip(icer, min_icer, max_icer)
    icercols = sc.vectocolor(icer, cmap=cmap1, midpoint=0)
    dalyscols = sc.vectocolor(da, cmap=cmap2, midpoint=0)
    
    ax1 = pl.subplot(2, 2, i+1)
    pl.scatter(cost[::stride], da[::stride], c=icercols[::stride], **skw)
    pl.ylabel('DALYs averted')
    
    # Cost effectiveness line
    pl.plot([0, cost.max()], [0, cost.max()/threshold], '--', **lkw)
    cmap = sc.orangebluecolormap()
    manualcolorbar(ax1, cmap1, vcenter=0, vmin=icer.min(), vmax=icer.max(), label=icerlabel)
    
    ax2 = pl.subplot(2,2, i+3)
    pl.scatter(cost[::stride], icer[::stride], c=dalyscols[::stride], **skw)
    pl.ylabel(icerlabel)
    
    # Cost effectiveness line
    pl.plot([0, cost.max()], [threshold]*2, '--', **lkw)
    manualcolorbar(ax2, cmap2, vcenter=0, vmin=da.min(), vmax=da.max(), label='DALYs averted')
    
    for j,ax in enumerate([ax1, ax2]):
        
        sc.commaticks(ax)
        ax.axhline(**lkw)
        ax.axvline(**lkw)
        ax.set_xlabel('Cost (US$)')
        ax.set_title(label)
        
        # Labels
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        str1 = 'Cost-\nsaving'
        str2 = ''#'Invalid (negative\nDALYs averted)'
        string1 = str1 if j == 0 else str2
        string2 = str2 if j == 0 else str1
        ax.text(x0/1.5, y1/2, string1)
        ax.text(x0/1.2, y0/2, string2)
        ax.text(x1/3, y0/2, str2)
        
        # Cost effective
        str1 = 'Cost-effective'
        str2 = 'Not cost-effective'
        pos1 = [x1/3, y1/2]
        pos2 = [x1/2, y1/50]
        string1 = str1 if j == 0 else str2
        string2 = str2 if j == 0 else str1
        ax.text(*pos1, string1)
        ax.text(*pos2, string2)
    
sc.figlayout()
pl.show()
if do_save:
    sc.savefig(f'{figstem}icers.png', dpi=100)
T.toc('Done')