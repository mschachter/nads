import time

import numpy as np
import matplotlib.pyplot as plt

import pyopencl as cl
from nads.gpu import GpuNetwork, ConstantInputStream
from nads.system import IFUnit

def test_if(nunits=10, sim_dur=0.500):

    ctx = cl.create_some_context()
    gpunet = GpuNetwork(ctx)

    instream = ConstantInputStream(1.25, 0.020, 0.150)
    gpunet.add_stream(instream)

    num_states = -1
    num_params = -1
    for k in range(nunits):
        u = IFUnit()
        num_states = u.num_states
        num_params = u.num_params
        gpunet.add_unit(u)

    u0 = gpunet.units[0]
    gpunet.connect_stream(instream, 0, u0, 1.0)

    for k,u in enumerate(gpunet.units[1:]):
        uprev = gpunet.units[k]
        w = 1.00 / float(k+2)
        #gpunet.connect_stream(instream, 0, u, w)
        gpunet.connect(uprev, u, 10.5)

    gpunet.compile()

    stime = time.time()
    step_size = 0.000250
    nsteps = int(sim_dur / step_size)
    all_states = []
    for k in range(nsteps):
        state = gpunet.step(step_size)
        #print 'k=%d, state=%s' % (k, str(state))
        all_states.append(state)
    etime = time.time() - stime
    print '%0.1fs to stimulate %0.3fs' % (etime, sim_dur)

    gpunet.clear()

    all_states = np.array(all_states)
    plt.figure()
    t = np.arange(0.0, sim_dur, step_size)
    for k in range(nunits):
        uindex = k*num_states
        v = all_states[:, uindex+1]
        #st = all_states[:, uindex]
        #print 'k=%d, # of spikes=%d' % (k, int(st.sum()))
        #v[st > 0.0] = 3.0

        plt.plot(t, v)
        #plt.plot(st, np.ones(len(st)), 'o')
    plt.axis('tight')

if __name__ == '__main__':
    test_if(nunits=1, sim_dur=0.020)

