import os
import sys
import traceback
import time
import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL.Image import open
import matplotlib.pyplot as plt

from math import pi, cos, sin

tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))


SCALE = 4
SIZE = 1000

class Visulizer(object):

    def __init__(self):
        # time.sleep(1.)
        # self.robot_listeners = {namespace: robot_listener(self, namespace, pose_type_string)\
		# 				 for namespace in all_robot_namespace}
        # self.subscription  # prevent unused variable warning
        # self.waypoints = np.asarray([[0.0,0.0]])
        # self.plot_time = 0.1
        # self.plot_timer = self.create_timer(self.plot_time, self.plot_callback)
        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # ax.scatter(0,0)
        # plt.show()
        self.i = 0
        # self._opengl_start()
        self.opengl_frame_rate = 30
        self._opengl_start()
        


    def _opengl_start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutInitWindowSize(SIZE,SIZE)
        glutInitWindowPosition(460, 0)
        glutCreateWindow('Distributed Bayesian Optimization')
        glutDisplayFunc(self.render)
        glutIdleFunc(self.render)
        # glutTimerFunc(20, self.render, 0)
        glutMainLoop()

    def _text(self, str, column, loc):
        if loc == 'left':
            glRasterPos3f(-1, 1.00 - 0.05 * column, 0.0)
        elif loc == 'right':
            glRasterPos3f(0.4, 1.00 - 0.05 * column, 0.0)
        n = len(str)
        for i in range(n):
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str[i]))

    def render(self, real_x=0, real_y=0, scale=SCALE, **kwargs):
        LOC_X = -real_x / scale
        LOC_Y = -real_y / scale
        glClearColor(0.753, 0.753, 0.753, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(LOC_X, LOC_Y, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # glEnable(GL_LINE_STIPPLE)
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)
        glVertex2f(-1., 0.)
        glVertex2f(1., 0.)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)
        glVertex2f(0., -1.)
        glVertex2f(0., 1.)
        glEnd()
        
        self._text(str(self.i), 1, 'left')
        ## plot axis

        def draw_circle(center, radius, side_num = 100, edge_only = True, color = 'w'):
            if(edge_only):
                glBegin(GL_LINE_LOOP)
            else:
                glBegin(GL_POLYGON)
            
            if color == 'o':
                glColor3f(1.0, 0.647, 0.0)
            elif color == 'y':
                glColor3f(1.0, 1.0, 0.878)
            elif color == 'blue':
                glColor3f(0.2, 0.3, 0.9)
            elif color == 'g':
                glColor3f(0.5, 1.0, 0.0)
            else:
                glColor3f(1.0, 1.0, 1.0)

                        
            for vertex in range(0, side_num):
                angle  = float(vertex) * 2.0 * pi / side_num
                glVertex2f(center[0] + cos(angle)*radius, center[1] +  sin(angle)*radius)
    
            glEnd()

        # robot_locs = [listener.get_latest_loc() for _, listener in self.robot_listeners.items()]
        robot_locs = [[0.,0.]]
        for loc in robot_locs:
            if loc is not None:
                draw_circle([l / SCALE for l in loc], radius=0.11 * 2.5 / SCALE, color='g')
                draw_circle([l / SCALE for l in loc], radius=0.05 / SCALE, edge_only = False, color='g')

        self.i += 1

        glutSwapBuffers()

        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)

def main_2():
    visulizer = Visulizer()


if __name__ == '__main__':
    main_2()
