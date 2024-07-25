import sciris as sc
import covasim as cv

class reset_rng(cv.Intervention):
    """ Reset the random number seed on each day """
    def apply(self, sim):
        seed = sim['rand_seed'] + sim.t
        try:
            sim.set_seed(seed=seed)
        except:
            print(f'Could not set seed {seed}')
        return 

sim_pars = sc.objdict(
    location  = 'malawi',
    # pop_type = 'hybrid', # Almost 2x slower
    pop_size  = 500_000, # Population size
    scaled_pop = 1.0e6, # Total population 1.0m
    pop_infected = 1000,
    rescale   = False,
    start_day = '2022-06-01', # First day of simulation
    end_day   = '2022-12-01', # Last day of simulation
    verbose   = 0.1, # How much detail to print (0 = none; -1 = one line per sim)
    rand_seed = 123456, # Random seed offset
)

people = None
sims = sc.autolist()
for i in range(10):
    sim = cv.Sim(sim_pars, rand_seed=i, people=people, interventions=reset_rng())
    if people is None:
        sim.init_people()
        people = sim.people
    sims += sim

with sc.timer():
    msim = cv.parallel(sims, keep_people=True)

msim.plot(max_sims=100)