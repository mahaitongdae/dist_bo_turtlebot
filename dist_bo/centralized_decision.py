# The centralized decision node. Handle the BO procedure.
import os
import sys
import traceback
import time
import json

import numpy as np
from argparse import Namespace

from functools import partial
from collections import deque
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray,Bool,String
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node

tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

from estimation.bayesian_optimization import BayesianOptimizationCentralized

from ros2_utils.benchmark_functions_2D import *
from ros2_utils.robot_listener import robot_listener
from std_srvs.srv import Trigger

function_dict = {'bird':Bird(), 'disk':Disk(), 'ackley': Ackley(), 'rosenbrock': Rosenbrock(),
                 'eggholder': Eggholder()}

class CentralizedDecision(Node):

    def __init__(self, pose_type_string, all_robot_namespace):
        super().__init__('bo_central')
        self.pose_type_string = pose_type_string
        self.robot_listeners = {namespace:robot_listener(self, namespace, self.pose_type_string, central=True)\
						 for namespace in all_robot_namespace}
        self.declare_parameter('function_type', 'ackley')
        self.declare_parameter('algorithm', 'ma_es')
        algo_args_dict = json.loads(open(tools_root + '/estimation/{}.json'.format(self.get_parameter('algorithm').value)).read())
        algo_args = Namespace(**algo_args_dict)

        self.func = function_dict[self.get_parameter('function_type').value].function
        
        self.obs = [None for i in range(len(all_robot_namespace)-1)]

        # publisher for each agent
        self.queries_publishers_ = [self.create_publisher(Float32MultiArray, '/{}/new_queries'.format(robot_namespace), 10) for robot_namespace in all_robot_namespace]
        # one publisher for queries
        # self.queries_publishers_ = self.create_publisher(Float32MultiArray, '/new_queries', 10)

        self.update_obs_time = 1.
        self.create_timer(self.update_obs_time, self.update_obs_callback)        
        
        if function_dict.get(algo_args.objective).arg_min is not None:
            arg_max = function_dict.get(algo_args.objective).arg_min
        else:
            arg_max = None

        algo_args.n_workers = len(all_robot_namespace)

        N = np.ones([algo_args.n_workers, algo_args.n_workers])

        self.bayesian_optimization_model = BayesianOptimizationCentralized(objective=function_dict.get(algo_args.objective),
                                                                            domain=function_dict.get(algo_args.objective).domain,
                                                                            arg_max=arg_max,
                                                                            n_workers=algo_args.n_workers,
                                                                            network=N,
                                                                            kernel='Matern',
                                                                            # length_scale_bounds=(1, 1000.0)
                                                                            acquisition_function=algo_args.acquisition_function,
                                                                            policy=algo_args.policy,
                                                                            fantasies=algo_args.fantasies,
                                                                            regularization=algo_args.regularization,
                                                                            regularization_strength=algo_args.regularization_strength,
                                                                            pending_regularization=algo_args.pending_regularization,
                                                                            pending_regularization_strength=algo_args.pending_regularization_strength,
                                                                            grid_density=algo_args.grid_density,
                                                                            args=algo_args)
        
        self.update_in_progress = False

    def update_obs_callback(self):
        targets_reached = [listener.is_target_reached() for _, listener in self.robot_listeners.items()]
        location = [listener.get_latest_loc() for _, listener in self.robot_listeners.items()]
        obs = [listener.get_observed_values() for _, listener in self.robot_listeners.items()]
        # self.get_logger().info(str(targets_reached))
        # self.get_logger().info(str(obs))
        if all(obs) and all(targets_reached) and not self.update_in_progress:
            self.update_in_progress = True
            self.get_logger().info('Start {} step BO'.format(self.bayesian_optimization_model.optimizer_step + 1))
            # if self.sim:
            #     self.obs = [listener.get_latest_readings() for _, listener in self.robot_listeners.items()]
            # else:
            #     self.obs = []
            # self.obs = [listener.get_observed_values() for _, listener in self.robot_listeners.items()]
            self.next_queries = self.bayesian_optimization_model.optimize(location, obs, plot=5)
            msg = Float32MultiArray()
            for i, publisher in enumerate(self.queries_publishers_):  
                flattern_q = self.next_queries[i].tolist()
                msg.data = flattern_q
                publisher.publish(msg)
            # msg.data = self.next_queries.reshape([-1]).tolist()
            # self.queries_publishers_.publish(msg)
            time.sleep(1.0) # wait the msg sent to robot nodes
            self.get_logger().info('Finish {} step BO'.format(self.bayesian_optimization_model.optimizer_step))
            self.update_in_progress = False
            for _, listener in self.robot_listeners.items():
                listener.reset_obs_target_stacks() 
        else:
            pass
    
    # def assign_targets_to_agents(self, targets, locations):
    #     locations = np.array(locations).reshape([-1, 2])
    #     inverted_locations = locations[::1]
    #     norm1 = 

def main(args=sys.argv):
    rclpy.init(args=args)
    pose_type = args[1]
    all_robots_namespace = args[2].split(',')

    centralized_decision = CentralizedDecision(pose_type, all_robots_namespace)

    rclpy.spin(centralized_decision)

    rclpy.shutdown()

if __name__ == '__main__':
    main()