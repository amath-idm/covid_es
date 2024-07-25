'''
Calculate Nepal ES costs -- from Mercy's email 2022-05-07
'''

import pylab as pl

x = [96, 135, 250] # Samples per month
y = [16_756, 20_122, 30_050] # Cost per month
dydx = (y[1]-y[0])/(x[1]-x[0]) # Linear so this works
y0 = y[0] - x[0]*dydx

print('ES cost calculator:')
print(f'y = {dydx:0.2f}x + {y0:0.2f}')

pl.plot(x,y, 'o-')
pl.xlabel('Number of samples per month')
pl.ylabel('Cost per month (USD)')
pl.show()