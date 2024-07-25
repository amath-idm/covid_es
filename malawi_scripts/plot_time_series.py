'''
Run a simple run to generate time series
'''

import sciris as sc
import covasim as cv
import covasim_es as ces
import sweep_grid as sg

rerun = False
filename = 'timeseries.exp'

if __name__ == '__main__':
    
    T = sc.timer()

    scenario = sg.make_scenario(debug=True)
    
    # Create, run, and export the experiment
    if rerun:
        exp = ces.Experiment(label='timeseries', n=5, scenario=scenario, sim_pars=sg.sim_pars)
        exp.run()
        sc.save(filename, exp)
    else:
        exp = sc.load(filename)

    # Analyze results
    print('Results:')
    print(exp.results)
    df = exp.results
    
    # Process as multisims
    with_es = [s for s in exp.msim.sims if s.scen.use_es==1]
    no_es   = [s for s in exp.msim.sims if s.scen.use_es==0]
    m_wes = cv.MultiSim(sims=with_es, label='With ES')
    m_nes = cv.MultiSim(sims=no_es, label='Without ES')
    m_wes.mean()
    m_nes.mean()
    msim = cv.MultiSim.merge(m_wes, m_nes, base=True)
    
    # Plot
    msim.plot(['cum_infections', 'cum_deaths'], show_args=dict(interventions=False))
    cv.savefig('covasim_es_timeseries.png')
    
    T.toc()
