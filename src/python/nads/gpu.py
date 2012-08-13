import copy
import os
import numpy as np

import pyopencl as cl

from nads.config import get_root_dir



KERNELS = {'TestUnit': read_cl(TestUnit.CL_FILE),
           'IFUnit': read_cl(IFUnit.CL_FILE)}


class GpuInputStream(object):

    def __init__(self):
        self.ndim = None

    def pull(self, t):
        return None

class GpuNetworkData(object):

    def __init__(self):

        self.unit2gpu = {}
        self.gpu2unit = {}

        self.unit_param_index = None
        self.params = None

        self.unit_state_index = None
        self.state = None
        self.next_state = None

        self.unit_weight_index = None
        self.weights = None
        self.conn_index = None
        self.num_connections = None

        self.unit_types = {}
        self.stream_uids = {}
        self.uids_stream = {}

        self.total_state_size = None


    def init_units(self, units, streams, stream_uids):

        self.streams = streams

        num_params = 0
        num_states = 0
        for u in units:
            num_params += len(u.params)
            num_states += len(u.state)

        self.stream_uids = stream_uids
        self.num_units = len(units)
        self.num_params = num_params
        self.num_states = num_states

        self.unit_state_index = np.zeros(self.num_units+len(self.stream_uids), dtype='int32')
        self.unit_param_index = np.zeros(self.num_units, dtype='int32')
        self.state = np.zeros(self.num_states+len(self.stream_uids), dtype='float32')
        self.params = np.zeros(self.num_params, dtype='float32')

        #assign states and parameters to the unit state and parameter arrays
        param_index = 0
        state_index = 0
        for k,u in enumerate(units):
            self.unit2gpu[u.id] = k
            self.gpu2unit[k] = u.id
            self.unit_state_index[k] = state_index
            self.unit_param_index[k] = param_index

            cname = u.__class__.__name__
            if cname not in self.unit_types:
                self.unit_types[cname] = []
            self.unit_types[cname].append(u.id)

            for m,s in enumerate(u.state):
                self.state[state_index+m] = s
            for m,pname in enumerate(u.param_order):
                self.params[param_index+m] = u.params[pname]

            state_index += len(u.state)
            param_index += len(u.params)

        #map the stream inputs to gpu indices
        for k,((stream_id,stream_index),suid) in enumerate(self.stream_uids.iteritems()):
            self.uids_stream[suid] = (stream_id,stream_index)
            gpu_index = self.num_units + k
            self.gpu2unit[gpu_index] = suid
            self.unit2gpu[suid] = gpu_index
            self.unit_state_index[gpu_index] = state_index
            state_index += 1

        self.total_state_size = state_index

        #set up an array to store the next states
        self.next_state = np.zeros(self.num_states)

    def init_weights(self, weights):

        weight_map = {}
        for (uid1,uid2),weight in weights.iteritems():
            if uid2 not in weight_map:
                weight_map[uid2] = []
            weight_map[uid2].append(uid1)

        total_num_conns = len(weights)
        self.num_connections = np.zeros(self.num_units, dtype='int32') #holds the number of input connections for each unit
        self.unit_weight_index = np.zeros(self.num_units, dtype='int32') #the index into the weight array for each unit
        self.weights = np.zeros(total_num_conns, dtype='float32') #the actual weights
        self.conn_index = np.zeros(total_num_conns, dtype='int32') #holds the indices of input connections for each unit
        weight_index = 0
        for uid,uconns in weight_map.iteritems():
            gpu_index = self.unit2gpu[uid]
            wlen = len(uconns)
            self.num_connections[gpu_index] = wlen
            self.unit_weight_index[gpu_index] = weight_index
            for m,pre_uid in enumerate(uconns):
                pre_gpu_index = self.unit2gpu[pre_uid]
                self.weights[weight_index+m] = weights[(pre_uid, uid)]
                self.conn_index[weight_index+m] = pre_gpu_index


            weight_index += wlen

    def copy_to_gpu(self, cl_context):

        mf = cl.mem_flags
        self.unit_param_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.unit_param_index)
        self.params_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.params)

        self.unit_state_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.unit_state_index)
        self.state_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.state)
        self.next_state_buf = cl.Buffer(cl_context, mf.WRITE_ONLY, self.next_state.nbytes)

        self.unit_weight_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.unit_weight_index)
        self.weights_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.weights)
        self.conn_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.conn_index)
        self.num_connections_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.num_connections)

    def update_state(self, cl_context, cl_queue, time):
        del self.next_state
        self.next_state = np.empty(self.num_states, dtype='float32')

        mf = cl.mem_flags
        cl.enqueue_copy(cl_queue, self.next_state, self.next_state_buf)

        self.state_buf.release()
        del self.state

        self.state = np.zeros([self.total_state_size], dtype='float32')
        self.state[:self.num_states] = self.next_state
        for s in self.streams:
            sval = s.pull(time)
            for sindex in range(s.ndim):
                skey = (s.id, sindex)
                suid = self.stream_uids[skey]
                gpu_index = self.unit2gpu[suid]
                state_index = self.unit_state_index[gpu_index]
                self.state[state_index] = sval[sindex]
        self.state_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.state)

    def clear(self):

        self.unit_param_index_buf.release()
        del self.unit_param_index_buf
        self.params_buf.release()
        del self.params_buf

        self.unit_state_index_buf.release()
        del self.unit_state_index_buf
        self.state_buf.release()
        del self.state_buf
        self.next_state_buf.release()
        del self.next_state_buf

        self.unit_weight_index_buf.release()
        del self.unit_weight_index_buf
        self.weights_buf.release()
        del self.weights_buf
        self.conn_index_buf.release()
        del self.conn_index_buf
        self.num_connections_buf.release()
        del self.num_connections_buf


class GpuNetwork(object):

    def __init__(self, cl_context):
        self.units = []
        self.connections = {}
        self.stream_connections = {}
        self.network_data = None
        self.cl_context = cl_context
        self.step_size_buf = None
        self.streams = []
        self.stream_uids = {}
        self.time = 0.0

    def add_stream(self, istream):
        self.streams.append(istream)
        stream_id = len(self.streams)-1
        istream.id = stream_id
        for k in range(istream.ndim):
            skey = (istream.id, k)
            uid = -(len(self.stream_uids)+1)
            self.stream_uids[skey] = uid

    def add_unit(self, u):
        uid = len(self.units)
        u.id = uid
        self.units.append(u)

    def connect(self, u1, u2, weight):
        ckey = (u1.id, u2.id)
        self.connections[ckey] = weight

    def connect_stream(self, stream, stream_index, unit, weight):
        """ Connect an element of an input stream to a unit """
        skey = (stream.id, stream_index)
        suid = self.stream_uids[skey]
        ckey = (suid, unit.id)
        self.connections[ckey] = weight

    def compile(self):
        self.network_data = GpuNetworkData()
        self.network_data.init_units(self.units, self.streams, self.stream_uids)
        self.network_data.init_weights(self.connections)

        if len(self.network_data.unit_types) > 1:
            print 'ERROR: only homogeneous unit types are currently allowed!'
            return

        self.kernel_cl = KERNELS[self.network_data.unit_types.keys()[0]]
        self.program = cl.Program(self.cl_context, self.kernel_cl).build()
        self.kernel = self.program.all_kernels()[0]

        self.queue = cl.CommandQueue(self.cl_context)

        self.network_data.copy_to_gpu(self.cl_context)

        print 'unit2gpu,',self.network_data.unit2gpu
        print 'gpu2unit,',self.network_data.gpu2unit

        print 'unit_param_index',self.network_data.unit_param_index
        print 'params',self.network_data.params

        print 'unit_state_index,',self.network_data.unit_state_index
        print 'state,',self.network_data.state
        print 'next_state,',self.network_data.next_state

        print 'unit_weight_index,',self.network_data.unit_weight_index
        print 'weights,',self.network_data.weights
        print 'conn_index,',self.network_data.conn_index
        print 'num_connections,',self.network_data.num_connections

    def step(self, step_size):

        global_size =  (len(self.units), )
        self.program.unit_step(self.queue, global_size, None,
                               self.network_data.unit_state_index_buf,
                               self.network_data.unit_param_index_buf,
                               self.network_data.state_buf,
                               self.network_data.params_buf,
                               self.network_data.unit_weight_index_buf,
                               self.network_data.conn_index_buf,
                               self.network_data.num_connections_buf,
                               self.network_data.weights_buf,
                               self.network_data.next_state_buf,
                               np.float32(step_size))

        self.time += step_size
        self.network_data.update_state(self.cl_context, self.queue, self.time)
        return self.network_data.state


    def clear(self):
        self.network_data.clear()



class ConstantInputStream(GpuInputStream):

    def __init__(self, amp, start, stop):
        GpuInputStream.__init__(self)
        self.ndim = 1
        self.start = start
        self.stop = stop
        self.amp = amp

    def pull(self, t):
        if t >= self.start and t < self.stop:
            return np.array([self.amp])
        else:
            return np.array([0.0])


