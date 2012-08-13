import time

import numpy as np
import matplotlib.pyplot as plt

import pyopencl as cl
from nads.gpu import GpuNetwork, ConstantInputStream
from nads.system import IFUnit, GpuUnit

class TestUnit(GpuUnit):
    CL_FILE = 'test.cl'

    def __init__(self):
        GpuUnit.__init__(self)

        self.param_order = ['a', 'b']
        self.params = {'a':1.0, 'b':5.0}
        self.state = [0.5, 0.75]


def test_basic():

    ctx = cl.create_some_context()

    gpunet = GpuNetwork(ctx)

    t1 = TestUnit()
    t1.state = [0.25, 0.5]
    t2 = TestUnit()
    t2.state = [0.7, 0.9]

    gpunet.add_unit(t1)
    gpunet.add_unit(t2)
    gpunet.connect(t1.id, t2.id, 0.5)

    gpunet.compile()

    nsteps = 5
    for k in range(nsteps):
        state = gpunet.step(0.0005)
        print 'k=%d, state=%s' % (k, str(state))

    gpunet.clear()


def test_if(nunits=10, sim_dur=0.500):

    ctx = cl.create_some_context()
    gpunet = GpuNetwork(ctx)

    instream = ConstantInputStream(1.00, 0.020, 0.150)
    gpunet.add_stream(instream)

    for k in range(nunits):
        u = IFUnit()
        gpunet.add_unit(u)

    u0 = gpunet.units[0]
    gpunet.connect_stream(instream, 0, u0, 2.75)

    for k,u in enumerate(gpunet.units[1:]):
        uprev = gpunet.units[k]
        #w = 1.00 / float(k+2)
        #gpunet.connect_stream(instream, 0, u, w)
        gpunet.connect(uprev, u, 20.0)

    gpunet.compile()

    stime = time.time()
    step_size = 0.00025
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
        uindex = k*2
        st = all_states[:, uindex]
        v = all_states[:, uindex+1]
        #print 'k=%d, # of spikes=%d' % (k, st.sum())
        v[st > 0.0] = 3.0

        plt.plot(t, v)
