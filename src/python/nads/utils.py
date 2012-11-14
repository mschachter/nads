import os

from OpenGL.GL import *
from OpenGL.GLU import *

import pyopencl as cl

from nads.config import get_root_dir

def read_cl(cl_name):
    cl_dir = os.path.join(get_root_dir(), 'src', 'cl')
    fname = os.path.join(cl_dir, cl_name)
    if not os.path.exists(fname):
        print 'Cannot open kernel .cl file: %s' % fname
        return ''
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


def gl_init(width, height):

    #glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)


    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, width/float(height), .1, 8192)
    #glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)

"""
def gl_lights():
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)

    light_position = [10., 10., 200., 0.]
    light_ambient = [.2, .2, .2, 1.]
    light_diffuse = [.6, .6, .6, 1.]
    light_specular = [2., 2., 2., 0.]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glEnable(GL_LIGHT0)

    mat_ambient = [.2, .2, 1.0, 1.0]
    mat_diffuse = [.2, .8, 1.0, 1.0]
    mat_specular = [1.0, 1.0, 1.0, 1.0]
    high_shininess = 3.

    mat_ambient_back = [.5, .2, .2, 1.0]
    mat_diffuse_back = [1.0, .2, .2, 1.0]

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    glMaterialfv(GL_BACK, GL_AMBIENT,   mat_ambient_back);
    glMaterialfv(GL_BACK, GL_DIFFUSE,   mat_diffuse_back);
    glMaterialfv(GL_BACK, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_BACK, GL_SHININESS, high_shininess);


"""

def gl_draw_line(v1, v2):
    glBegin(GL_LINES)
    glVertex3f(v1[0], v1[1], v1[2])
    glVertex3f(v2[0], v2[1], v2[2])
    glEnd()


def gl_draw_axes():
    #X Axis
    glColor3f(1,0,0)    #red
    v1 = [0,0,0]
    v2 = [1,0,0]
    gl_draw_line(v1, v2)

    #Y Axis
    glColor3f(0,1,0)    #green
    v1 = [0,0,0]
    v2 = [0,1,0]
    gl_draw_line(v1, v2)

    #Z Axis
    glColor3f(0,0,1)    #blue
    v1 = [0,0,0]
    v2 = [0,0,1]
    gl_draw_line(v1, v2)

