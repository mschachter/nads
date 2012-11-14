import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties

import numpy as np

from nads.utils import gl_draw_axes

class NetworkWindow(object):

    def __init__(self, network, step_size=0.00025):
        """
            Creates a visualization window for a given network. The network cannot be compiled yet!
        """

        self.network = network

        #set up timing parameters
        self.refresh_rate_ms = 30 #in ms
        self.sim_dur_per_refresh = 0.005 # amount of time
        self.step_size = step_size
        self.num_steps_per_refresh = int(self.sim_dur_per_refresh / self.step_size)

        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = np.array([0., 0.])
        self.rotate = np.array([0., 0., 0.])
        self.translate = np.array([0., 0., 0.])
        self.initrans = np.array([0., 0., -2.])

        self.width = 640
        self.height = 480

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow('Network')

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)


        #this will call draw every 30 ms
        glutTimerFunc(self.refresh_rate_ms, self.timer, self.refresh_rate_ms)

        #setup OpenGL scene
        self.glinit()

        #create position and color VBOs
        unit_positions = self.network.get_unit_positions()
        print 'unit_positions:'
        print unit_positions
        self.pos_vbo = vbo.VBO(data=unit_positions, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.pos_vbo.bind()

        unit_colors = self.network.get_unit_colors()
        print 'unit_colors:'
        print unit_colors
        self.col_vbo = vbo.VBO(data=unit_colors, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.col_vbo.bind()

        self.network.visualize = True
        self.network.color_vbo = self.col_vbo

        #create a CL context for the network to use
        plats = cl.get_platforms()
        if sys.platform == "darwin":
            cl_context = cl.Context(properties=get_gl_sharing_context_properties(), devices=[])
        else:
            props = [(cl.context_properties.PLATFORM, plats[0])] + get_gl_sharing_context_properties()
            cl_context = cl.Context(properties=props, devices=None)

        self.network.cl_context = cl_context

        #compile the network
        self.network.compile()

        glutMainLoop()


    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #gluPerspective(60., self.width / float(self.height), .1, 1000.)
        gluPerspective(60., self.width / float(self.height), .1, 10.)
        glMatrixMode(GL_MODELVIEW)


    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            sys.exit()
        elif args[0] == 't':
            pass

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old[:] = [x, y]


    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old[0]
        dy = y - self.mouse_old[1]
        if self.mouse_down and self.button == 0: #left button
            self.rotate[0] += dy * .2
            self.rotate[1] += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate[2] -= dy * .01
        self.mouse_old[:] = [x, y]


    def draw(self):

        #run network for a few iterations
        for k in range(self.num_steps_per_refresh):
            #print 'Step %d... step size=%0.6f' % (k, self.step_size)
            self.network.step(self.step_size)

        #render
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans[0], self.initrans[1], self.initrans[2])
        glRotatef(self.rotate[0], 1, 0, 0)
        glRotatef(self.rotate[1], 0, 1, 0) #we switched around the axis so make this rotate_z
        glTranslatef(self.translate[0], self.translate[1], self.translate[2])

        #render the network
        glEnable(GL_POINT_SMOOTH)
        glPointSize(5)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)

        self.pos_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, self.pos_vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_POINTS, 0, len(self.network.units))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)

        #draw the x, y and z axis as lines
        gl_draw_axes()

        glutSwapBuffers()
