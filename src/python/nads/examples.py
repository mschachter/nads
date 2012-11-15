import time

import numpy as np
import matplotlib.pyplot as plt

import pyopencl as cl
from nads.gpu import GpuNetwork, ConstantInputStream, RandomInputStream
from nads.system import IFUnit
from nads.ui import NetworkWindow


def create_cube_if_net(width, height, depth):

    gpunet = GpuNetwork()

    max_cube_width = 1.0
    mcw_half = max_cube_width / 2.0

    dx = max_cube_width / width
    dy = max_cube_width / height
    dz = max_cube_width / depth

    max_len = max_cube_width*np.sqrt( np.sqrt(2*max_cube_width**2)  ) #longest distance in cube between two neurons

    all_units = np.ndarray([width, height, depth], dtype='object')

    #build network
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                u = IFUnit()
                u.position = np.array([mcw_half - x*dx, mcw_half - y*dy, mcw_half - z*dz])
                gpunet.add_unit(u)
                all_units[x, y, z] = u

    #connect units topographically
    print 'Connecting Neurons...'
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                u = all_units[x, y, z]
                for u2 in all_units.ravel():
                    d = np.linalg.norm(u.position - u2.position)
                    p_connect = min(1.0, 1.0 - (d / max_len)*0.2)
                    w = max(np.random.randn(), -0.25)
                    print 'd=%0.6f, p_connect=%0.6f, w=%0.6f' % (d, p_connect, w)
                    if np.random.rand() < p_connect:
                        gpunet.connect(u, u2, np.random.randn())

    #create input streams
    nstreams = 10
    all_streams = []
    for k in range(nstreams):
        rs = RandomInputStream(0.0, 1.5, 0.050, 1000.00, positive=True)
        gpunet.add_stream(rs)
        all_streams.append(rs)

    #connect inputs up
    max_inputs = 5
    for u in all_units[:, :, 0].ravel():
        nin = np.random.randint(max_inputs+1)
        streams_connected = {}
        for k in range(nin):
            rsnum = np.random.randint(nstreams)
            if rsnum not in streams_connected:
                gpunet.connect_stream(all_streams[rsnum], 0, u, 1.0)
                streams_connected[rsnum] = True

    return gpunet


def test_cube_gl(nunits=5, step_size=0.000250):

    gpunet = create_cube_if_net(nunits, nunits, nunits)
    nw = NetworkWindow(gpunet, step_size=step_size)


def create_chain_if_net(nunits):

    gpunet = GpuNetwork()

    instream = ConstantInputStream(1.25, 0.020, 1.0)
    gpunet.add_stream(instream)

    num_states = -1
    num_params = -1
    dx = 0.05
    dy = 5e-6
    dz = 5e-6
    for k in range(nunits):
        u = IFUnit()
        u.position = np.array([0.0, k*dx, 0.0])

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

    return gpunet,num_states,num_params


def test_if(nunits=10, sim_dur=0.500):

    gpunet,num_states,num_params = create_chain_if_net(nunits)

    gpunet.compile()

    stime = time.time()
    step_size = 0.00250
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


def test_if_gl(nunits=10, step_size=0.000250):

    gpunet,num_states,num_params = create_chain_if_net(nunits)
    nw = NetworkWindow(gpunet, step_size=step_size)


if __name__ == '__main__':
    test_if(nunits=1, sim_dur=0.020)

