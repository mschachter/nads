from inspect import getmembers

import numpy as np

class Param(object):
    """ Encodes a dynamical system parameter. Parameters aren't supposed to change. """
    def __init__(self, shape, default_value):
        self.shape = shape
        self.default_value = default_value

class State(object):
    """ Encodes a dynamical system state, which is subject to change and can also have it's history stored on the GPU. """
    def __init__(self, shape, num_lags=0, default_value=0.0):
        self.shape = shape
        self.num_lags = num_lags
        self.default_value = default_value

class System(object):
    """ Base class for a discrete dynamical system that's encoded to run on a GPU via an OpenCL kernel. """

    def __init__(self, params, initial_state):
        self.id = None
        self.params = []       #the parameters of the system
        self.state = []        #the state of the system
        self.state_offset = {} #key is state name, value is start index of state value in self.state
        self.param_offset = {} #key is param name, value is start index of param value in self.params

        #initialize parameters and states
        for (name,obj) in getmembers(self):
            tname = type(obj).__name__
            if tname == 'Param':
                self.handle_param(name, obj, params)
            elif tname == 'State':
                self.handle_state(name, obj, initial_state)

    def check_shape(self, shape1, shape2):
        if len(shape1) != len(shape2):
            return False
        for s1,s2 in zip(shape1, shape2):
            if s1 != s2:
                return False
        return True

    def handle_element(self, name, obj, default_params, offset_dict, obj_array):

        tname = type(obj).__name__
        #get default param value (or one specified in default_params)
        if name in default_params:
            dval = default_params[name]
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

        #put the parameter values into self.params
        offset_dict[name] = len(obj_array)
        nlags = 0
        if tname == 'State':
            nlags = obj.num_lags

        for k in range(nlags+1):
            for val in dval.ravel():
                obj_array.append(val)

    def handle_param(self, name, param, default_params):
        self.handle_element(name, param, default_params, self.param_offset, self.params)


    def handle_state(self, name, state, initial_state):
        self.handle_element(name, state, initial_state, self.state_offset, self.state)

    def generate_opencl(self, output_file):
        pass



class IFUnit(System):

    R = Param(shape=[1], default_value=1.0)
    C = Param(shape=[1], default_value=1e-2)
    vthresh = Param(shape=[1], default_value=1.0)
    vreset = Param(shape=[1], default_value=0.0)

    v = State(shape=[1], num_lags=0, default_value=0.0)
    spike_time = State(shape=[1], num_lags=5, default_value=-1.0)

    def __init__(self, params=dict(), initial_state=dict()):
        System.__init__(self, params=params, initial_state=initial_state)

