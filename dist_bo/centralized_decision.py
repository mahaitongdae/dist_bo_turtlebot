# The centralized decision node. Handle the BO procedure.
import os
import sys
import traceback
import time

import numpy as np
import argparse

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

    def __init__(self):
        super().__init__('bo_central')

        self.declare_parameter('function_type', 'ackley')

        self.declare_parameter('algorithm', 'UCB-ES')

        self.func = function_dict[self.get_parameter('function_type').value].function

        neighborhood_namespaces = set(['MobileSensor1'])

        self.robot_listeners = {namespace:robot_listener(self,namespace,self.pose_type_string)\
						 for namespace in neighborhood_namespaces}
        
        self.obs = [None for i in range(len(neighborhood_namespaces)-1)]

        self.queries_publisher_ = self.create_publisher(Float32MultiArray, 'queries', 10)

        self.update_obs_time = 1.
        self.create_timer(self.update_obs_time, self.update_obs_callback)

        parser = argparse.ArgumentParser()
        parser.add_argument('--objective', type=str, default='ackley')
        parser.add_argument('--constraint', type=str, default='disk')
        parser.add_argument('--model', type=str, default='torch') #torch or sklearn
        # parser.add_argument('--arg_max', type=np.ndarray, default=None)
        parser.add_argument('--n_workers', type=int, default=10)
        parser.add_argument('--kernel', type=str, default='Matern')
        parser.add_argument('--acquisition_function', type=str, default='ucbpe')
        parser.add_argument('--policy', type=str, default='greedy')
        parser.add_argument('--unconstrained', type=bool, default=True)
        parser.add_argument('--decision_type', type=str, default='parallel')
        parser.add_argument('--fantasies', type=int, default=0)
        parser.add_argument('--regularization', type=str, default=None)
        parser.add_argument('--regularization_strength', type=float, default=0.01)
        parser.add_argument('--pending_regularization', type=str, default=None)
        parser.add_argument('--pending_regularization_strength', type=float, default=0.01)
        parser.add_argument('--grid_density', type=int, default=30)
        parser.add_argument('--n_iters', type=int, default=10)
        parser.add_argument('--n_runs', type=int, default=5) # TODO: change parser to ros parameters.
        args = parser.parse_args()
        if args.n_workers == 3:
            N = np.ones([3,3])
            # N[0, 1] = N[1, 0] = N[1, 2] = N[2, 1] = 1
            # args.n_iters = 50
        elif args.n_workers == 1:
            N = np.ones([1, 1])
            args.n_iters = 150
        else:
            N = np.ones([args.n_workers,args.n_workers])
            # assert args.n_workers == N.shape[0]
            if function_dict.get(args.objective).arg_min is not None:
                arg_max = function_dict.get(args.objective).arg_min
            else:
                arg_max = None

        self.bayesian_optimization_model = BayesianOptimizationCentralized(objective=function_dict.get(args.objective),
                                                                            domain=function_dict.get(args.objective).domain,
                                                                            arg_max=arg_max,
                                                                            n_workers=args.n_workers,
                                                                            network=N,
                                                                            kernel='Matern',
                                                                            # length_scale_bounds=(1, 1000.0) remove this greatly improve performance?
                                                                            acquisition_function=args.acquisition_function,
                                                                            policy=args.policy,
                                                                            fantasies=args.fantasies,
                                                                            regularization=args.regularization,
                                                                            regularization_strength=args.regularization_strength,
                                                                            pending_regularization=args.pending_regularization,
                                                                            pending_regularization_strength=args.pending_regularization_strength,
                                                                            grid_density=args.grid_density,
                                                                            args=args)

    def update_obs_callback(self):
        targets_reached = [listener.if_target_reached for listener in self.robot_listeners]
        if any(targets_reached):
            self.obs = [listener.get_latest_readings() for listener in self.robot_listeners]
            self.next_queries = self.bayesian_optimization_model.optimize(self.obs)
            msg = Float32MultiArray()
            msg.data = list(self.next_queries)
            self.queries_publisher_.pub(self.next_queries)
        else:
            pass