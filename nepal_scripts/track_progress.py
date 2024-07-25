import os
import numpy as np
import sciris as sc
import pylab as pl

sleep = 10
do_plot = False
folder = '.'

while True:
    
    sc.heading(f'\nChecking progress at {sc.getdate()}...')
    
    files = sc.getfilelist(folder, '*.tmp')
    nfiles = len(files)

    if nfiles:
        
        last = int(files[0].split('_')[-1][:-4]) # E.g. pull out 15360 from sim_0289_of_15360.tmp
        
        times = sc.autolist()
        for f in files:
            times += os.path.getctime(f)
        
        times.sort()
        times = np.array(times)
        times -= times[0]
        now = times[-1]
        avtime = now/nfiles
        endtime = last*avtime
        nremaining = last - nfiles
        remaining = endtime - now
        
        sc.progressbar(nfiles, last, length=100, newline=True)
        print(f'Files: {nfiles:n} done, {last:n} total, {nremaining:n} remaining')
        print(f'Time: {now:n} s done, {endtime:n} s total, {remaining:n} s remaining, {avtime:n} s average')
        
        if do_plot:
            pl.scatter(times, np.arange(nfiles))
            pl.scatter(endtime, last)
        
    else:
        print('No files found')
    
    
    sc.timedsleep(sleep)
