import os
import sys
import traceback
import time
import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL.Image import open
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray,Bool,String
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node

SCALE = 60
SIZE = 1000


class Visulizer(Node):

    def __init__(self,):
        super().__init__(node_name = 'visulization', namespace = 'distributed_exploration') #todo:check namespace
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/MobileSensor1/waypoints',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.waypoints = np.asarray([[0.0,0.0]])
        self.plot_time = 0.1
        self.plot_timer = self.create_timer(self.plot_time, self.plot_callback)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # ax.scatter(0,0)
        # plt.show()

    def listener_callback(self, msg):
        self.get_logger().info('msg data type "%s"' % type(msg.data))
        self.get_logger().info('msg data len "%s"' % len(msg.data))
        self.waypoints = np.reshape(np.asarray(msg.data),[-1, 2])
    
    def plot_callback(self):
        try:
            self.ax.plot(self.waypoints[:,0],self.waypoints[:,1])
            # my_loc = self.get_my_loc()
            # print(my_loc,type(my_loc))
            # self.ax.scatter(my_loc[0],my_loc[1],c='Red',s=20)
            plt.xlim(0, 1.2)
            plt.ylim(-3, 0)
            plt.pause(0.05)
            plt.cla()
        except:
            pass


    def _opengl_start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutInitWindowSize(SIZE,SIZE)
        glutInitWindowPosition(460, 0)
        glutCreateWindow('Distributed Bayesian Optimization')
        glutDisplayFunc(self.render)
        glutIdleFunc(self.render)
        # glutTimerFunc(20,self.render, 0)
        glutMainLoop()

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


def main(args=None):
    rclpy.init(args=args)

    visulizer = Visulizer()

    rclpy.spin(visulizer)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    visulizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
