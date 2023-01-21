import sys

from example_interfaces.srv import AddTwoInts
from custom_interfaces.srv import Query2DFunc
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(Query2DFunc, '/query_2d_func')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Query2DFunc.Request()

    def send_request(self):
        self.req.x = 0.0
        self.req.y = 1.0
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        self.get_logger().info(
        'Result of query new function: for %.3f + %.3f = %.3f' %
        (0.0, 0.0, self.future.result().obj))
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request()
    minimal_client.get_logger().info(
        'Result of query new function: for %.3f + %.3f = %.3f' %
        (0.0, 0.0, response.obj))

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()