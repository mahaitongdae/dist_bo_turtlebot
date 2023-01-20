import os
import sys

import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from std_msgs.msg import String, Float32
from custom_interfaces.srv import Query2DFunc

tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

from ros2_utils.benchmark_functions_2D import *

function_dict = {'bird':Bird(), 'disk':Disk(), 'ackley': Ackley(), 'rosenbrock': Rosenbrock(),
                 'eggholder': Eggholder()}

class VirtualSourceByFunc(Node):
    def __init__(self):
        super().__init__('virtual_source')
        self.declare_parameter('function_type', 'ackley')
        self.func = function_dict[self.get_parameter('function_type').value].function
        self.srv = self.create_service(Query2DFunc, 'query_2d_func', self.query_callback)
        
    def query_callback(self, request, response):
        loc = (request.x, request.y)
        response.obj = self.func(loc)
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.x, request.y))
        return response

def main(args=None):
    rclpy.init(args=args)

    virtual_source_func = VirtualSourceByFunc()

    rclpy.spin(virtual_source_func)

    rclpy.shutdown()

if __name__ == '__main__':
    main()

