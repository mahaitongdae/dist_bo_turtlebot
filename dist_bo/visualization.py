import os
import sys
import traceback
import time
import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL.Image import open
import matplotlib.pyplot as plt

from example_interfaces.srv import AddTwoInts

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray,Bool,String
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from math import pi, cos, sin
from collections import deque

tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

from ros2_utils.robot_listener import robot_listener
SCALE = 4
SIZE = 800
INITLOC = (700, 0)

class Visulizer(object):

    def __init__(self):
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
        self.robot_locs = [None]


    def listener_callback(self, msg):
        self.get_logger().info('msg data type "%s"' % type(msg.data))
        self.get_logger().info('msg data len "%s"' % len(msg.data))
        self.waypoints = np.reshape(np.asarray(msg.data),[-1, 2])

    def run(self):
        time.sleep(1.)
        self._opengl_start()


    def _opengl_start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutInitWindowSize(SIZE,SIZE)
        glutInitWindowPosition(INITLOC[0], INITLOC[1])
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

        # robot_locs = [[0.,0.]]
        for loc in self.robot_locs:
            if loc is not None:
                draw_circle([l / SCALE for l in loc], radius=0.11 * 2.5 / SCALE, color='g')
                draw_circle([l / SCALE for l in loc], radius=0.05 / SCALE, edge_only = False, color='g')
                # self.get_logger().info('draw robot at {}, {}'.format(loc[0] / SCALE, loc[1]/ scale))


        self.i += 1

        glutSwapBuffers()

        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)

    def set_robot_loc(self, robot_locs):
        self.robot_locs = robot_locs

class VisulizerNode(Node):

    def __init__(self, pose_type_string, all_robot_namespace, center=(0., 0.), scale=4.):
        super().__init__('visulization')
        self.robot_listeners = {namespace: robot_listener(self, namespace, pose_type_string, visualizer=True)\
						 for namespace in all_robot_namespace}
        qos = QoSProfile(depth=10)
        self.central_listeners = self.create_subscription(Float32MultiArray, '/bo_central/to_visualize', self.central_listeners_cbk, qos)
        self.amaxucb_loc = None

        self.i = 0
        self.opengl_frame_rate = 30
        self.vis_fps = 30
        self.create_timer(1 / self.vis_fps, self.update_loc_cbk)
        self.robot_locs = [None]   
        self.robot_targets = [None]
        self.robot_look_ahead_targets = [None]
        self.center = center
        self.get_logger().info('center is {}'.format(self.center))
        self.scale = scale
        
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()
        self.run()

    def central_listeners_cbk(self, msg):
        self.amaxucb_loc = msg.data

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def listener_callback(self, msg):
        self.get_logger().info('msg data type "%s"' % type(msg.data))
        self.get_logger().info('msg data len "%s"' % len(msg.data))
        self.waypoints = np.reshape(np.asarray(msg.data),[-1, 2])

    def run(self):
        time.sleep(1.)
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

    def loc2screen(self, loc):
        return (loc[0] - self.center[0]) / self.scale, (loc[1] - self.center[1]) / self.scale

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

        response = self.send_request(self.i, 0)
        
        self._text(str(response.sum) + ' frames', 1, 'left')
        # plot axis

        def get_gl_color(color):
            if color == 'o':
                glColor3f(1.0, 0.647, 0.0)
            elif color == 'y':
                glColor3f(1.0, 1.0, 0.878)
            elif color == 'blue':
                glColor3f(0.2, 0.3, 0.9)
            elif color == 'g':
                glColor3f(0.5, 1.0, 0.0)
            elif color == 'r' or color == 'red':
                glColor3f(1.0, 0.05, 0.05)
            elif color == 'lr' or color == 'lighter_red':
                glColor3f(1.0, 0.5, 0.5)
            else:
                glColor3f(1.0, 1.0, 1.0)

        def draw_circle(center, radius, side_num = 100, edge_only = True, color = 'w'):
            center = self.loc2screen(center)
            radius = radius / self.scale
            if(edge_only):
                glBegin(GL_LINE_LOOP)
            else:
                glBegin(GL_POLYGON)
            get_gl_color(color)                        
            for vertex in range(0, side_num):
                angle  = float(vertex) * 2.0 * pi / side_num
                glVertex2f(center[0] + cos(angle) * radius, center[1] +  sin(angle)*radius)
    
            glEnd()

        def draw_arrow(center, direction, len = 0.5, color='w'):
            glBegin(GL_LINES)
            get_gl_color(color)
            glVertex2f(center[0], center[1])
            glVertex2f(center[0] + len * np.cos(direction), center[1] + len * np.sin(direction))
            glEnd()
            glBegin(GL_LINES)
            get_gl_color(color)
            glVertex2f(center[0] + len * np.cos(direction), center[1] + len * np.sin(direction))
            glVertex2f(center[0] + len * np.cos(direction) + 0.3 * len * np.cos(direction + 5 / 6 * np.pi), center[1] + len * np.sin(direction) + 0.3 * len * np.sin(direction + 5 / 6 * np.pi))
            glEnd()
            glBegin(GL_LINES)
            get_gl_color(color)
            glVertex2f(center[0], center[1])
            glVertex2f(center[0] + len * np.cos(direction), center[1] + len * np.sin(direction))
            glVertex2f(center[0] + len * np.cos(direction) + 0.3 * len * np.cos(direction + 7 / 6 * np.pi), center[1] + len * np.sin(direction) + 0.3 * len * np.sin(direction + 7 / 6 * np.pi))
            glEnd()

        
                
        def draw_light(center, color = 'o', size=1.):
            draw_circle(center, size * 0.1, edge_only=False, color=color)
            center_in_screen = self.loc2screen(center)
            for direction in np.linspace(0, 2 * np.pi, 8):
                for lower_direction in np.linspace(direction-0.01, direction+0.01, 20):
                    glBegin(GL_LINES)
                    get_gl_color(color=color)
                    glVertex2f(center_in_screen[0] + size * 0.04 * np.cos(lower_direction), center_in_screen[1] + 0.04 * np.sin(lower_direction))
                    glVertex2f(center_in_screen[0] + size * 0.06 * np.cos(lower_direction), center_in_screen[1] + 0.06 * np.sin(lower_direction))
                    glEnd()

        # estimated source loc
        if self.amaxucb_loc is not None:
                # draw_light(self.amaxucb_loc, color='o')
                self._text('estimate maximum: ({:.3f}, {:.3f})'.format(self.amaxucb_loc[0], self.amaxucb_loc[1]), 2, 'left')

        # draw_light([0., 0.], color='red')
        draw_light([-1.25, 2.24], color='red')
        draw_light([-1.25, 1.66], color='lr', size=0.8)
        draw_light([-1.55, 1.96], color='lr', size=0.8)
        draw_light([-0.85, 1.96], color='lr', size=0.8)
        
        # robot_locs = [[0.,0.]]
        for i, loc in enumerate(self.robot_locs):
            if loc is not None:
                draw_circle(loc, radius=0.11 * 2.5, color='blue')
                draw_circle(loc, radius=0.05, edge_only = False, color='blue')
                # self.get_logger().info('draw robot at {}, {}'.format(loc[0] / SCALE, loc[1]/ scale))
                if self.robot_yaw[i] is not None:
                    draw_arrow(self.loc2screen(loc), self.robot_yaw[i], len=0.11 * 2.5 / SCALE, color='blue')
                

        for i, loc in enumerate(self.robot_targets):
            if loc is not None:
                # draw_circle([l / SCALE for l in loc], radius=0.11 * 2.5 / SCALE, color='g')
                draw_circle(loc, radius=0.05, edge_only = False, color='g')
                # self.get_logger().info('draw robot at {}, {}'.format(loc[0] / SCALE, loc[1]/ scale))
                if self.robot_locs[i] is not None:
                    glBegin(GL_LINES)
                    glColor3f(1.0, 1.0, 1.0)
                    glVertex2f(*self.loc2screen(self.robot_locs[i]))
                    glVertex2f(*self.loc2screen(self.robot_targets[i]))
                    glEnd()

        for i, loc in enumerate(self.robot_look_ahead_targets):
            if loc is not None:
                # draw_circle([l / SCALE for l in loc], radius=0.11 * 2.5 / SCALE, color='g')
                draw_circle(loc, radius=0.05, edge_only = False, color='y')


        # self.get_logger().info('loc array {}'.format(self.robot_locs))

        self.i += 1

        glutSwapBuffers()

        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)

    def update_loc_cbk(self):
        self.robot_locs = [listener.get_latest_loc() for _, listener in self.robot_listeners.items()]
        self.robot_yaw = [listener.get_latest_yaw() for _, listener in self.robot_listeners.items()]
        self.robot_targets = [listener.get_new_queries() for _, listener in self.robot_listeners.items()]
        self.robot_look_ahead_targets = [listener.get_look_ahead_target() for _, listener in self.robot_listeners.items()]
        
        # self.get_logger().info("update robot locs")

    # def set_robot_loc(self, robot_locs):
    #     self.robot_locs = robot_locs


def main(args=sys.argv):

    rclpy.init(args=args)
    pose_type = args[1]
    all_robots_namespace = args[2].split(',')
    is_real_exp = int(args[3])
    center = (-1.5, 2.0) if is_real_exp else (0., 0.)
    visulizer = VisulizerNode(pose_type, all_robots_namespace, center=center)

    # future = visulizer.run()

    rclpy.spin_once(visulizer)

    rclpy.shutdown()