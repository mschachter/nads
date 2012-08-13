import os
from nads.config import get_root_dir

import pyopencl as cl

def read_cl(cl_name):
    cl_dir = os.path.join(get_root_dir(), 'src', 'cl')
    fname = os.path.join(cl_dir, cl_name)
    f = open(fname, 'r')
    cl_str = f.read()
    f.close()
    return cl_str


def print_device_info():

    ctx = cl.create_some_context()
    devices = ctx.get_info(cl.context_info.DEVICES)
    device = devices[0]

    print 'Vendor: %s' % device.vendor
    print 'Name: %s' % device.name
    print 'Max Clock Freq: %0.0f' % device.max_clock_frequency
    gmem = float(device.global_mem_size) / 1024**2
    print 'Global Memory: %0.0f MB' % gmem
    print '# of Compute Units: %d' % device.max_compute_units
