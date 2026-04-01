"""
Class for generating stochastic transcriptional dynamics
"""

import numpy as np
from numba import njit


#Right now, this is just a bundle of functions
class snapshot_simulator:
    
    def __init__(self):
        pass
    
    #Next: include other models
    def simulate(self,kplus,kminus,rburst,D,T,dt,z,bndry_func):
        Tbirth_sd,Nbirths_sd,class_index_sd = self.population_dynamics(z=z,kplus=kplus,kminus=kminus,rburst=rburst,T=T)
        return self.spatial_dynamics(Tbirth_sd, Nbirths_sd, class_index_sd, D, T, dt, z, bndry_func)
    

    def simulate_nospace(self,kplus,kminus,rburst,T,num_obs):
        z = np.zeros(num_obs)
        _,counts,_ = self.population_dynamics(z=z,kplus=kplus,kminus=kminus,rburst=rburst,T=T)
        return counts


    #population dynamics. Right now. I'm only doing the bursty expression model
    def population_dynamics(self,z=None,kplus=None,kminus=None,rburst=None,T=None,kon=None,koff=None):
        sample_size = z.shape[0]
        #Birth events, Births per event
        Nbirthevents = np.random.poisson(kplus*T,size=sample_size)
        Nbirths = np.random.geometric(1/rburst, size = np.sum(Nbirthevents))
        #Total births per snapshot, total births across all snapshots
        A = np.repeat(np.arange(0,sample_size), Nbirthevents)
        Nbirthstotal_vec = np.array([np.sum(Nbirths[i == A]) for i in np.arange(sample_size)])
        Nbirthstotal = np.sum(Nbirths)
        #Times of events, extended for each birth
        Tbirthevents = np.random.uniform(0,T,size = np.sum(Nbirthevents))
        Tbirth = np.repeat(Tbirthevents, Nbirths)
        #times of death per birth, which ones survive
        tdecay = np.random.exponential(scale = 1/kminus, size = Nbirthstotal)
        Tdecay = Tbirth + tdecay
        survived_degradation = Tdecay > T
        #ID for each particle in each snapshot that survived, their times, births
        class_index = np.repeat(np.arange(0, sample_size), Nbirthstotal_vec)
        class_index_sd = class_index[survived_degradation]
        Tbirth_sd = Tbirth[survived_degradation]
        Nbirths_sd = np.bincount(class_index_sd, minlength=sample_size)
        return Tbirth_sd,Nbirths_sd,class_index_sd
        
    ### spatial dynamics: 
    # Tbirth_sd = times of birth for particles that survived degradation (sd)
    # Nbirths_sd = number of particle births for each birth time (sd)
    # class_index_sd = A label tha tells us what snapshot the particle is in (assumes sd)
    def spatial_dynamics(self,Tbirth_sd,Nbirths_sd,class_index_sd,D,T,dt,z,bndry_func):
        #Early output if nothing survives degradation
        sample_size = z.shape[0]
        if Tbirth_sd.size == 0:
            empty = np.array([])
            return [empty for i in range(sample_size)]
        sz_sd = len(Tbirth_sd)
        time = Tbirth_sd
        pos = np.repeat(z, Nbirths_sd, axis=0)
        time_counter = np.min(time)
        sd_index = np.arange(sz_sd)
        tracking_index = sd_index
        survived_boundary = bndry_func(pos, class_index_sd)
        v = np.sqrt(2 * D * dt)

        for _ in range(int(np.ceil((T - time_counter) / dt))):
            # Update dynamics
            time[tracking_index] += dt
            pos[tracking_index] = np.add(pos[tracking_index],v*np.random.normal(0,1,pos[tracking_index].shape))

            survived_boundary[tracking_index] = bndry_func(pos[tracking_index], class_index_sd[tracking_index])
            tracking_index = sd_index[(survived_boundary & (time < T))]

        # Remaining particles after time T
        sb = survived_boundary & (time >= T)
        class_index_survived = class_index_sd[sb]
        pos_survived = pos[sb]

        # Group positions by class index
        data = [pos_survived[class_index_survived == c] for c in range(sample_size)]
        
        return data

