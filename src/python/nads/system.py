import operator
from inspect import getmembers

import numpy as np

class Param(object):
    """ Encodes a dynamical system parameter. Parameters aren't supposed to change. """
    def __init__(self, shape, default_value, order=-1):
        self.shape = shape
        self.default_value = default_value
        self.order = order

class State(object):
    """ Encodes a dynamical system state, which is subject to change and can also have it's history stored on the GPU. """
    def __init__(self, shape, num_lags=0, default_value=0.0, order=-1):
        self.shape = shape
        self.num_lags = num_lags
        self.default_value = default_value
        self.order = order

class System(object):
    """ Base class for a discrete dynamical system that's encoded to run on a GPU via an OpenCL kernel. """

    def __init__(self, initial_params, initial_state):
        self.id = None
        self.params = {}       #the parameters of the system
        self.state = {}        #the state of the system

        state_order = []
        param_order = []

        #initialize parameters and states
        for (name,obj) in getmembers(self):
            tname = type(obj).__name__
            if tname == 'Param':
                self.handle_param(name, obj, initial_params)
                param_order.append( (obj.order, name, np.sum(obj.shape)) )
            elif tname == 'State':
                self.handle_state(name, obj, initial_state)
                state_order.append( (obj.order, name, np.sum(obj.shape)*(obj.num_lags+1)) )

        #get sorted order of state and params
        state_order.sort(key=operator.itemgetter(0))
        param_order.sort(key=operator.itemgetter(0))

        self.num_states = np.sum([x[2] for x in state_order])
        self.num_params = np.sum([x[2] for x in param_order])

        self.state_order = [x[1] for x in state_order]
        self.param_order = [x[1] for x in param_order]


    def check_shape(self, shape1, shape2):
        if len(shape1) != len(shape2):
            return False
        for s1,s2 in zip(shape1, shape2):
            if s1 != s2:
                return False
        return True

    def handle_element(self, name, obj, obj_map, default_values):

        tname = type(obj).__name__
        #get default param value (or one specified in default_params)
        if name in default_values:
            dval = default_values[name]
        else:
            dval = obj.default_value

        #only deal with numpy arrays
        if np.isscalar(dval):
            dval = np.array([dval])
        else:
            dval = np.array(dval)

        #check the shape
        dshape = dval.shape
        if not self.check_shape(dshape, obj.shape):
            print 'Invalid shape of default element value specified for %s %s, should be %s but is %s' %\
                  (tname, name, str(obj.shape), str(dshape))

        #create a vector of the appropriate dimensions
        nlags = 0
        if tname == 'State':
            nlags = obj.num_lags
        vshape = [nlags+1]
        vshape.extend(obj.shape)
        #print 'element=%s, shape=%s, dval=%s' % (name, str(vshape), str(dval))
        v = np.ndarray(vshape)

        for k in range(nlags+1):
            v[k, :] = dval

        #set the state/param
        obj_map[name] = v


    def handle_param(self, name, param, default_params):
        self.handle_element(name, param, self.params, default_params)


    def handle_state(self, name, state, initial_state):
        self.handle_element(name, state, self.state, initial_state)

    def generate_opencl(self, output_file):
        pass

    def get_state_vector(self):
        s = []
        for name in self.state_order:
            sval = self.state[name]
            s.extend(sval.ravel())
        return s

    def get_param_vector(self):
        p = []
        for name in self.param_order:
            pval = self.params[name]
            p.extend(pval.ravel())
        return p

    def get_kernel_name(self):
        pass


class IFUnit(System):

    R = Param(shape=[1], default_value=1.0, order=0)
    C = Param(shape=[1], default_value=1e-2, order=1)
    vthresh = Param(shape=[1], default_value=1.0, order=2)
    vreset = Param(shape=[1], default_value=0.0, order=3)
    synapse_tau = Param(shape=[1], default_value=0.010, order=4)

    output = State(shape=[1], num_lags=0, default_value=0.0, order=0)
    v = State(shape=[1], num_lags=0, default_value=0.0, order=1)
    spike_time = State(shape=[1], num_lags=0, default_value=0.0, order=2)


    def get_kernel_name(self):
        return 'integrate_and_fire.cl'

    def __init__(self, initial_params=dict(), initial_state=dict()):
        System.__init__(self, initial_params=initial_params, initial_state=initial_state)
        self.position = [0.0, 0.0, 0.0]

