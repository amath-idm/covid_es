import numpy as np
import pylab as pl


x = [0, 12, 48]
y = [6175, 8272]
y.insert(0, y[0] - (y[1]-y[0])/3)

es_min = y[0]
es_sample = (y[-1]-es_min)/x[-1]

print(es_min, '+', es_sample)

pl.plot(x,y)