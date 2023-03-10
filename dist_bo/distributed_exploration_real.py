# The distributed exploration node. This reduces the number of topic subscription needed.
import os
import sys
import traceback
import time
import datetime

import numpy as np
import pandas as pd

from functools import partial
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import patches

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray,Bool,String,Float32
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node

# from custom_interfaces.srv import Query2DFunc
import math

tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

# General dependencies
from ros2_utils.robot_listener import robot_listener
from ros2_utils.pose import prompt_pose_type_string
from ros2_utils.misc import get_sensor_names

# Estimation dependencies
from estimation.ConsensusEKF import ConsensusEKF 
from util_func import joint_meas_func, analytic_dhdz, top_n_mean

# Waypoint planning dependencies
from motion_control import WaypointPlanning
from motion_control.SimplePositionControl import Turtlebot3Path
from motion_control.WaypointTracking import BURGER_MAX_LIN_VEL

from util_func import analytic_dLdp, joint_F_single
from consensus import consensus_handler

# Motion control dependencies
from ros2_utils.pose import bounded_change_update, turtlebot_twist, stop_twist

from motion_control.WaypointTracking import LQR_for_motion_mimicry

from collision_avoidance.obstacle_detector import obstacle_detector, source_contact_detector, boundary_detector
from collision_avoidance.regions import RegionsIntersection, CircleInterior
from ros2_utils.benchmark_functions_2D import *

function_dict = {'bird':Bird(), 'disk':Disk(), 'ackley': Ackley(), 'rosenbrock': Rosenbrock(),
                 'eggholder': Eggholder(), 'fakeackley': fakeAckley()}


COEF_NAMES = ['C1','C0','k','b']
VISUALIZE = False
REACH_DIST = 1.

def get_control_action(waypoints,curr_x):
	if len(waypoints)==0:
		return []
		
	planning_dt = 0.1

	Q = np.array([[10,0,0],[0,10,0],[0,0,1]])
	R = np.array([[10,0],[0,1]])

	uhat,_,_ = LQR_for_motion_mimicry(waypoints,planning_dt,curr_x,Q=Q,R=R)

	return uhat

def truncate_angle(angle):
	while True:
		if angle > math.pi:
			angle = angle - 2 * math.pi
		elif angle < - math.pi:
			angle = angle + 2 * math.pi
		else:
			break
	return angle

class distributed_seeking(Node):
	def __init__(self, robot_namespace, pose_type_string, init_target_loc, data_dir=None, neighborhood_namespaces=None, xlims=[-np.inf,np.inf], ylims = [-np.inf,np.inf]):
		super().__init__(node_name = 'distributed_seeking', namespace = robot_namespace)
		self.pose_type_string = pose_type_string
		self.robot_namespace = robot_namespace

		assert(robot_namespace in neighborhood_namespaces)
		if neighborhood_namespaces is None:
			self.neighborhood_namespaces = get_sensor_names(self)
		else:
			self.neighborhood_namespaces = neighborhood_namespaces

		self.robot_listeners = {namespace:robot_listener(self,namespace,self.pose_type_string)\
						 for namespace in neighborhood_namespaces}

		self.id = int(robot_namespace[-1])
		if self.id > 1:
			self.collsion_avoidance_neighbors = ['MobileSensor{}'.format(i + 1) for i in range(self.id - 1)]
		else:
			self.collsion_avoidance_neighbors = None
		self.target_reached = False
		self.position_control_step = 1
		self.main_loop_step = 1
		self.func = function_dict['fakeackley'].function
		self.other_robot_locs = None
		self._DT_ = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
		if data_dir is None:
			self._ROOT_DIR_ = '/home/mht/turtlebot3_ws/src/dist_bo/results'
			self._TEMP_DIR_ = os.path.join(self._ROOT_DIR_, 'real')
			self._ID_DIR_ = os.path.join(self._TEMP_DIR_, self._DT_)
			self._DATA_DIR_ = os.path.join(self._ID_DIR_, "data")
		else:
			self._DATA_DIR_ = data_dir
		self._FIG_DIR_ = os.path.join(self._DATA_DIR_, "figures")
		self._PNG_DIR_ = os.path.join(self._FIG_DIR_, "png")
		self._PDF_DIR_ = os.path.join(self._FIG_DIR_, "pdf")

		for dir in [self._DATA_DIR_, self._FIG_DIR_, self._PNG_DIR_, self._PDF_DIR_]:
			os.makedirs(dir, exist_ok=True)	

		qos = QoSProfile(depth=10)

		"""
		Timer initialization
		"""

		# self.waypoint_sleep_time = 0.5
		# self.waypoint_timer = self.create_timer(self.waypoint_sleep_time,self.simple_waypoint_callback)

		# self.motion_sleep_time = 0.1
		# self.motion_timer = self.create_timer(self.motion_sleep_time,self.simple_position_callback)

		self.record_time = 0.2
		self.create_timer(self.record_time, self.record_loc_cbk)

		if VISUALIZE:
			self.plot_time = 0.1
			self.plot_timer = self.create_timer(self.plot_time, self.plot_callback)
			plt.ion()
			self.fig, self.ax = plt.subplots()
				

		""" 
		Waypoint planning initializations 
		"""
			
		# # Temporary hard-coded waypoints used in devel.	
		self.waypoints = np.array([[1.,-2.5],[2.,-1.5],[3,-0.5]])
		# # self.waypoints = np.array([[3,-0.5]])
		self.target_loc = np.array(init_target_loc)
		self.look_ahead_target_loc = self.target_loc.copy()
		self.get_logger().info('Initial target location of {}: ({:.1f}, {:.1f})'.format(robot_namespace, self.target_loc[0], self.target_loc[1]))
	
		"""
		Motion control initializations
		"""

		self.MOVE = False
		self.move_sub = self.create_subscription(Bool,'/MISSION_CONTROL/MOVE',self.MOVE_CALLBACK, qos)
	
		self.vel_pub = self.create_publisher(Twist, '/{}/cmd_vel'.format(self.robot_namespace), qos)

		self.control_actions = deque([])
		self.controller = Turtlebot3Path()

		# Obstacles are expected to be circular ones, parametrized by (loc,radius)
		self.obstacle_detector = obstacle_detector(self)
		self.source_contact_detector = source_contact_detector(self)
		self.boundary_detector = boundary_detector(self,xlims,ylims)

		self.FIRST_CONTACT_ROBOT = None
		self.SOURCE_LOC = None

		# current control actions
		self.v = 0.0
		self.omega = 0.0

		# receiving next-query
		qos = QoSProfile(depth=10)
		self.create_subscription(Float32MultiArray, '/{}/new_queries'.format(robot_namespace), self.new_query_callback, qos)
		# self.create_subscription(Float32MultiArray, '/new_queries'.format(robot_namespace), self.new_query_callback, qos)

		# pub observation
		self.obs = 0.0
		self.observe_publisher = self.create_publisher(Float32, '/{}/observation'.format(self.robot_namespace), qos)
		# self.client = self.create_client(Query2DFunc, '/query_2d_func')
		self.target_reached_publisher = self.create_publisher(Bool, '/{}/target_reached'.format(self.robot_namespace), qos)
		self.look_ahead_publisher = self.create_publisher(Float32MultiArray, '/{}/look_ahead'.format(self.robot_namespace), qos)
		

		# while not self.client.wait_for_service(timeout_sec=1.0):
		# 	self.get_logger().info('service not available, waiting again...')
		# self.req = Query2DFunc.Request()

		self.main_loop_freq = 10
		self.create_timer(float(1/self.main_loop_freq), self.main_loop_callback)

		self.main_loop_counter = 0

	def main_loop_callback(self):
		'''
		main loop steps
		1. move to target
		2. query source & publish observation
		3. 
		'''

		if self.MOVE:
		
			if self.main_loop_step == 1:
				self.simple_position_callback()
			
			elif self.main_loop_step == 2:
				self.vel_pub.publish(stop_twist())
				self.send_query()
				self.publish_obs_callback()
			
			elif self.main_loop_step == 3:
				self.vel_pub.publish(stop_twist())
			
			msg = Bool()
			msg.data = self.target_reached
			self.target_reached_publisher.publish(msg)
			# self.publish_look_ahead_target_cbk()
			self.main_loop_counter += 1
			self.main_loop_counter = self.main_loop_counter % self.main_loop_freq
		
		else:
			self.vel_pub.publish(stop_twist())

	def send_query(self):
		while True:
			try:
				self.obs = self.robot_listeners[self.robot_namespace].get_latest_readings()[0]
				break
			except:
				self.get_logger().info('get obs: {}, type {}'.format(self.robot_listeners[self.robot_namespace].get_latest_readings(), self.robot_namespace))
		# loc = self.get_my_loc()
		# self.obs = -1.0 * self.func(loc)

	def query_virtual_source_callback(self):
		if self.target_reached:
			self.send_query()
		else:
			pass

	def central_all_obs_received_callback(self, msg):
		self.central_all_obs_received = msg.data

	def publish_obs_callback(self):
		self.get_logger().info('subscriber number {}'.format(self.observe_publisher.get_subscription_count()))
		if self.observe_publisher.get_subscription_count():
			msg = Float32()
			msg.data = self.obs
			self.observe_publisher.publish(msg)
			self.get_logger().info('Obs published!')
			self.main_loop_step += 1
		else:
			pass

	def publish_look_ahead_target_cbk(self):
		if self.look_ahead_publisher.get_subscription_count():
			msg = Float32MultiArray()
			msg.data = self.look_ahead_target_loc
			self.look_ahead_publisher.publish(msg)
			
	# def publish_target_reached_callback(self):
	# 	msg = Bool()
	# 	msg.data = self.target_reached
	# 	self.target_reached_publisher.publish(msg)

	def new_query_callback(self, data):
		if self.main_loop_step != 3:
			# in the step of obs published
			pass
		else:
			time.sleep(0.5) # wait for other qu
			other_target_locs = []
			neighborhood_namespaces = self.neighborhood_namespaces.copy()
			neighborhood_namespaces.remove(self.robot_namespace)
			if 'Source0' in neighborhood_namespaces:
				neighborhood_namespaces.remove('Source0')
			for rbt_namespace in neighborhood_namespaces:
				while True:
					other_query = self.robot_listeners[rbt_namespace].get_new_queries()
					if other_query is not None:
						self.get_logger().info('Get {}\'s q: {}'.format(rbt_namespace, other_query))
						other_target_locs.append(other_query)
						break
					else:
						self.get_logger().info(rbt_namespace)
						time.sleep(0.1)
			free_space = RegionsIntersection(self.obstacle_detector.get_free_spaces(origins=other_target_locs) + self.boundary_detector.get_free_spaces())
			self.target_loc = free_space.project_point(np.asarray([data.data[0], data.data[1]])).squeeze()
			self.get_logger().info('Get new target loc: ({:.3f}, {:.3f})'.format(self.target_loc[0], self.target_loc[1]))
			self.main_loop_step = 1 # reset main loop to new step
			self.target_reached = False

	def project_target_loc(self, my_loc):
		look_ahead_dist = 0.3
		look_ahead_space = CircleInterior(my_loc, look_ahead_dist)
		self.look_ahead_target_loc = look_ahead_space.project_point(self.target_loc)
		other_robot_locs = []
		neighborhood_namespaces = self.neighborhood_namespaces.copy()
		neighborhood_namespaces.remove(self.robot_namespace)
		for rbt_namespace in neighborhood_namespaces:
			neighbor_loc = self.robot_listeners[rbt_namespace].get_latest_loc()
			if neighbor_loc is not None:
				other_robot_locs.append(neighbor_loc)
		self.other_robot_locs = other_robot_locs.copy()
		if len(other_robot_locs) > 0:
			free_space = RegionsIntersection(self.obstacle_detector.get_free_spaces(origins=other_robot_locs))
			self.look_ahead_target_loc = free_space.project_point_sideway(self.look_ahead_target_loc, my_loc)

	def FIRST_FOUND_callback_(self,data):
		
		self.FIRST_CONTACT_ROBOT = data.data

	def SOURCE_LOC_callback_(self,data):
		
		self.SOURCE_LOC = data.data

	def check_source_found_(self):

		# Check if this robot has become the FIRST_CONTACT_ROBOT
		if self.SOURCE_LOC is None and self.FIRST_CONTACT_ROBOT is None:
			if self.source_contact_detector.contact():
				self.SOURCE_LOC = self.source_contact_detector.get_source_loc()
				self.FIRST_CONTACT_ROBOT = self.robot_namespace

		# The following being true means this robot has found the source and is the first one to do so.
		# Only the FIRST_CONTACT ROBOT has the right to publish the source location. This is to avoid network jamming.
		if (not self.SOURCE_LOC is None) and self.FIRST_CONTACT_ROBOT == self.robot_namespace: 
			
			out = Float32MultiArray()
			out.data = list(np.array(self.SOURCE_LOC).ravel().astype(float))
			self.SOURCE_LOC_pub.publish(out)

			out = String()
			out.data = self.robot_namespace
			self.FIRST_FOUND_pub.publish(out)

	def est_reset(self):
		self.estimator.reset()
		self.q_hat = self.estimator.get_q()
	
	def waypoint_reset(self):
		self.waypoints = []
		self.FIM = np.ones((2,2))*1e-4
		self.F = 0
		self.cons.reset()
	
	def motion_reset(self):
		self.control_actions = deque([])
		self.v = 0.0
		self.omega = 0.0

	def list_coefs(self,coef_dicts):
		if len(coef_dicts)==0:
			# Hard-coded values used in development.	
			C1=-0.3
			C0=0
			b=-2
			k=1
		else:
			C1=np.array([v['C1'] for v in coef_dicts])
			C0=np.array([v['C0'] for v in coef_dicts])
			b=np.array([v['b'] for v in coef_dicts])
			k=np.array([v['k'] for v in coef_dicts])
		return C1,C0,b,k

	def neighbor_h(self,coef_dicts=[]):
		C1,C0,b,k = self.list_coefs(coef_dicts)
			
		return partial(joint_meas_func,C1,C0,k,b)
	
	def neighbor_dhdz(self,coef_dicts=[]):
		C1,C0,b,k = self.list_coefs(coef_dicts)

		return partial(analytic_dhdz,C1s = C1,C0s = C0,ks = k,bs = b)

	def neighbor_zhats(self):

		return self.nb_zhats

	def z_hat_callback(self,data,namespace):
		self.nb_zhats[namespace] = np.array(data.data).flatten()

	def process_readings(self,readings):
		return top_n_mean(np.array(readings),4)

	def dLdp(self,q_hat,ps,FIM,coef_dicts=[]):

		C1,C0,b,k = self.list_coefs(coef_dicts)
	
		return analytic_dLdp(q=q_hat,ps=ps, C1s = C1, C0s = C0, ks=k, bs=b,FIM=FIM)
	
	def get_my_loc(self):
		return self.robot_listeners[self.robot_namespace].get_latest_loc()
	
	def get_my_yaw(self):
		return self.robot_listeners[self.robot_namespace].get_latest_yaw()
	
	def get_my_coefs(self):
		return self.robot_listeners[self.robot_namespace].get_coefs()

	def calc_new_F(self,coef_dicts=[]):

		C1,C0,b,k = self.list_coefs(coef_dicts)

		return	joint_F_single(qhat=self.q_hat,ps=self.get_my_loc().reshape(1,-1),C1 = C1, C0 = C0, k=k, b =b)

	def MOVE_CALLBACK(self,data):

		if not self.MOVE == data.data:
			if data.data:
				self.get_logger().info('Robot Moving')
			else:
				self.get_logger().info('Robot Stopping')

		self.MOVE = data.data

	def est_callback(self):
		""" 
				Estimation 
		"""
		if self.MOVE:
		# if True:
				p = []
				y = []
				zhat = []
				coefs = []

				zh = self.estimator.get_z()

				for name,sl in self.robot_listeners.items():
					# self.get_logger().info('scalar y:{}.'.format(self.process_readings(sl.get_latest_readings())))
					# print(name,sl.coef_future.done(),sl.get_coefs())	
					loc = sl.get_latest_loc()
					reading = sl.get_latest_readings()
					coef = sl.get_coefs()

					# self.get_logger().info('name:{} loc:{} reading:{} coef:{}'.format(name,loc, reading,coef))
					if (not loc is None) and \
						 (not reading is None) and\
						 	len(coef)==len(COEF_NAMES):
							p.append(loc)
							y.append(self.process_readings(reading))
							# print(self.process_readings(sl.get_latest_readings()),sl.get_latest_readings())
						
							if len(self.nb_zhats[name])>0:
								zhat.append(self.nb_zhats[name])
							else:
								zhat.append(zh)

							coefs.append(coef)

				zhat = np.array(zhat)

				# self.get_logger().info('zhat:{}. zh:{} y:{} p:{} coefs:{}'.format(zhat,zh,y,p,coefs))
				try:
					if not self.source_contact_detector.contact(): # Freeze the estimator update after source contact.
					# if True:
						if len(p)>0 and len(y)>0 and len(zhat)>0:
							self.estimator.update(self.neighbor_h(coefs),self.neighbor_dhdz(coefs),y,p,zhat\
												,z_neighbor_bar=None,consensus_weights=self.consensus_weights(y,p))

						# Publish z_hat and q_hat
						z_out = Float32MultiArray()
						z_out.data = list(zh)
						self.z_hat_pub.publish(z_out)

						qh = self.estimator.get_q()
						q_out = Float32MultiArray()
						q_out.data = list(qh)
						self.q_hat_pub.publish(q_out)
						self.get_logger().info('qhat:{}'.format(qh))
						self.q_hat = qh 
					else:
						self.est_reset()

				except ValueError as err:
					self.get_logger().info("Not updating due to ValueError")
					traceback.print_exc()
		else:
			self.est_reset()

	def simple_waypoint_callback(self):
		"""
			Simplified waypoint tracking for testing the new bayesian optimization codes.
		"""
		if self.MOVE:
			my_loc = self.get_my_loc()
			my_coefs = self.get_my_coefs()
			self.target_loc = np.array([1.,-2.5])
			self.waypoints = WaypointPlanning.straight_line(self.target_loc,my_loc\
														,planning_horizon = 20\
														,step_size = self.waypoint_sleep_time * BURGER_MAX_LIN_VEL)	

	def waypoint_callback(self):
		"""
			Waypoint Planning
		"""
		if self.MOVE:
		# if True:
			if self.source_contact_detector.contact():
			# if False:
				self.waypoints = []
			else:
				my_loc = self.get_my_loc()
				my_coefs = self.get_my_coefs()
				free_space = RegionsIntersection(self.obstacle_detector.get_free_spaces() + self.boundary_detector.get_free_spaces() )

				target_loc = None
				if (not my_loc is None) and len(my_coefs)>0:
					if not self.SOURCE_LOC is None:
						target_loc = self.SOURCE_LOC
						self.waypoints = WaypointPlanning.straight_line(target_loc,my_loc\
																	,planning_horizon = 20\
																	,step_size = self.waypoint_sleep_time * BURGER_MAX_LIN_VEL)	
			
					elif len(self.q_hat)>0:
						target_loc = RegionsIntersection(self.boundary_detector.get_free_spaces()).project_point(self.q_hat)
						# target_loc = self.q_hat

						# Get neighborhood locations and coefs, in matching order.
						# Make sure my_coef is on the top.

						neighbor_loc = []

						neighborhood_coefs =[]
						for name,nl in self.robot_listeners.items():
							if not name == self.robot_namespace:
								loc = nl.get_latest_loc()
								coef = nl.get_coefs()
								if (not loc is None) and (len(coef)>0):
									neighbor_loc.append(loc)
									neighborhood_coefs.append(coef)
						
						neighborhood_coefs = [self.get_my_coefs()]+neighborhood_coefs # Make sure my_coef is on the top.

					
						self.waypoints = WaypointPlanning.waypoints(target_loc,my_loc,neighbor_loc,lambda qhat,ps: self.dLdp(qhat,ps,FIM=self.FIM,coef_dicts = neighborhood_coefs), \
																	step_size = self.waypoint_sleep_time * BURGER_MAX_LIN_VEL\
																		,planning_horizon = 20\
																		,free_space=free_space)	
						# Note the FIM arg in self.dLdp is set to be self.FIM, which is the consensus est. of global FIM.

				
			# self.get_logger().info("Current Waypoints:{}".format(self.waypoints))
				
		else:
			self.waypoint_reset()

	def FIM_consensus_callback(self):
		if self.MOVE:
		# if True:
		# Consensus on the global FIM estimate.
			if (not self.get_my_loc() is None):
				newF = self.calc_new_F()
				dF = newF-self.F
				self.cons.timer_callback(dx=dF) # Publish dF to the network.
				self.FIM = self.cons.get_consensus_val().reshape(self.FIM.shape)
				self.F = newF

	def simple_motion_callback(self):
		loc = self.get_my_loc()
		yaw = self.get_my_yaw()
		# self.get_logger().info('loc:{} yaw:{}'.format(loc,yaw))
		# dist_to_target_loc = np.linalg.norm(loc - self.target_loc, 2)
		# reach_target = dist_to_target_loc < REACH_DIST

		if len(self.waypoints)==0:
			self.get_logger().info("Running out of waypoints.")

		if (not loc is None) and (not yaw is None) and len(self.waypoints)>0:
			curr_x = np.array([loc[0],loc[1],yaw])		
			self.control_actions = deque(get_control_action(self.waypoints,curr_x))

			# self.get_logger().info("Waypoints:{}".format(self.waypoints))
			wp_proj = self.waypoints
			waypoint_out = Float32MultiArray()
			waypoint_out.data = list(wp_proj.flatten())
			self.waypoint_pub.publish(waypoint_out)

		if len(self.control_actions)>0:
			# Pop and publish the left-most control action.
			[v,omega] = self.control_actions.popleft()
			[v,omega] = bounded_change_update(v,omega,self.v,self.omega) # Get a vel_msg that satisfies max change and max value constraints.
			vel_msg = turtlebot_twist(v,omega)

			# Update current v and omega
			self.v = v
			self.omega = omega
			self.vel_pub.publish(vel_msg)
			# self.get_logger().info('{}'.format(vel_msg))
		else:
			self.vel_pub.publish(stop_twist())

			# Update current v and omega
			self.v = 0.0
			self.omega = 0.0

	def simple_position_callback(self):
		twist = Twist()
		loc = self.get_my_loc()
		yaw = self.get_my_yaw()
		
		if (loc is not None) and (yaw is not None):
			distance = math.sqrt(
						(self.target_loc[1] - loc[1])**2 +
						(self.target_loc[0] - loc[0])**2)
			if distance <= 0.03 and self.position_control_step == 1:
				self.get_logger().info("initial state reach target")
				time.sleep(0.5) # in case other nodes are not ready
				self.position_control_step = 3
				return

			if self.target_reached is True:
				self.get_logger().info('Target reached!')
				self.vel_pub.publish(stop_twist())
			else:
				path_theta = math.atan2(
						self.target_loc[1] - loc[1],
						self.target_loc[0] - loc[0])
				angle = path_theta - yaw #TODO:check the coord system in real experiments
				
				# Step 1: Turn
				if self.position_control_step == 1:
					angular_velocity = 1.0  # unit: rad/s
					# self.get_logger().info('path_theta: {:.3f}, self yaw = {:.3f}'.format(path_theta, yaw))
					twist, self.position_control_step = self.controller.turn(angle, angular_velocity, self.position_control_step)

				# Step 2: Go Straight
				elif self.position_control_step == 2:
					linear_velocity = 0.1  # unit: m/s
					angular_velocity = 0.4  # unit: rad/s

					self.project_target_loc(loc)
					distance = math.sqrt(
						(self.look_ahead_target_loc[1] - loc[1])**2 +
						(self.look_ahead_target_loc[0] - loc[0])**2)
					path_theta = math.atan2(
						self.look_ahead_target_loc[1] - loc[1],
						self.look_ahead_target_loc[0] - loc[0])
					angle = truncate_angle(path_theta - yaw)
					# if self.id == 1:
					# 	self.get_logger().info('distance: {:.3f}, path_theta:{:.3f}, yaw: {:.3f}, angle: {:.3f}'.format(distance, path_theta, yaw, angle))
						# self.get_logger().info('look ahead loc: {}'.format(self.look_ahead_target_loc))
					if abs(angle) > 0.5 * math.pi:
						twist = stop_twist()
						self.position_control_step = 1
					else:
						twist, self.position_control_step = self.controller.go_straight(distance, angle, linear_velocity, angular_velocity, self.position_control_step)
					# self.get_logger().info('distance: {:.3f}, angle: {:.3f}, output_turn: {:.3f}'.format(distance, angle, twist.angular.z))
				# # Step 3: Turn
				# elif self.step == 3:
				# 	angle = self.goal_pose_theta - self.last_pose_theta
				# 	angular_velocity = 0.1  # unit: rad/s

				# 	twist, self.step = Turtlebot3Path.turn(angle, angular_velocity, self.step)

				# Reset
				elif self.position_control_step == 3:
					self.position_control_step = 1
					self.target_reached = True
					self.get_logger().info('Target reached!')
					self.main_loop_step += 1
					time.sleep(0.5)
				self.vel_pub.publish(twist)

	def record_loc_cbk(self):
		loc = self.get_my_loc()
		yaw = self.get_my_yaw()
		if loc is not None and yaw is not None:
			data = {'time': time.time(),
					'{}_x1'.format(self.robot_namespace):[loc[0]],
	   				'{}_x2'.format(self.robot_namespace):[loc[1]],
					'{}_yaw'.format(self.robot_namespace):[yaw],}
			df = pd.DataFrame().from_dict(data)
			filepath = os.path.join(self._DATA_DIR_,'pose.csv')
			if 'pose.csv' not in os.listdir(self._DATA_DIR_):
				df.to_csv(filepath)
			else:
				df.to_csv(filepath, mode='a', header=False)  
		


	def plot_callback(self):
		# try:
		self.ax.scatter(self.look_ahead_target_loc[0], self.look_ahead_target_loc[1],c='blue',s=20)
		self.ax.scatter(self.target_loc[0], self.target_loc[1],c='green',s=20)
		my_loc = self.get_my_loc()
		my_yaw = self.get_my_yaw()
		if my_loc is not None and my_yaw is not None:
			self.ax.arrow(my_loc[0],my_loc[1],0.2 * np.cos(my_yaw), 0.2 * np.sin(my_yaw))
			self.ax.scatter(my_loc[0],my_loc[1],c='Red',s=20)
		def plot_others(loc):
			self.ax.add_patch(patches.Circle(loc, 0.11 * 2.5, fc = 'none', ec = 'red'))
		if self.other_robot_locs is not None:
			for loc in self.other_robot_locs:
				if loc is not None:
					plot_others(loc)
		plt.axis('equal')
		plt.xlim(-3, 3)
		plt.ylim(-3, 3)
		plt.title(self.robot_namespace)
		plt.pause(0.05)
		plt.cla()
		# except:
		# 	pass
			

	def query_source_callback(self, data):
		query_points = data.data
		self.target_loc = (query_points[2 * (self.robot_id-1), 2 * (self.robot_id-1) + 1])
		self.get_logger().info('Get new source at ({}, {})'.format(query_points[2 * (self.robot_id-1), 2 * (self.robot_id-1) + 1]))
		self.target_reached = False
		self.get_logger().info('Reset target reached: {}'.format(str(self.target_reached)))

	def motion_callback(self):
		"""
			Motion Control
		"""
		if self.MOVE:
		# if True:
			if self.source_contact_detector.contact():
			# if False:
				self.vel_pub.publish(stop_twist())
				# self.get_logger().info('Source Contact')
			else:

				# Project waypoints onto obstacle-free spaces.
				
				free_space = RegionsIntersection(self.obstacle_detector.get_free_spaces() + self.boundary_detector.get_free_spaces() )

				loc = self.get_my_loc()
				yaw = self.get_my_yaw()
				# self.get_logger().info('loc:{} yaw:{}'.format(loc,yaw))

				if len(self.waypoints)==0:
					self.get_logger().info("Running out of waypoints.")

				if (not loc is None) and (not yaw is None) and len(self.waypoints)>0:
					curr_x = np.array([loc[0],loc[1],yaw])		
					wp_proj = free_space.project_point(self.waypoints)
					self.control_actions = deque(get_control_action(wp_proj,curr_x))

					# self.get_logger().info("Waypoints:{}".format(self.waypoints))
					waypoint_out = Float32MultiArray()
					waypoint_out.data = list(wp_proj.flatten())
					self.waypoint_pub.publish(waypoint_out)

				if len(self.control_actions)>0:

					# Pop and publish the left-most control action.

					[v,omega] = self.control_actions.popleft()
					
					[v,omega] = bounded_change_update(v,omega,self.v,self.omega) # Get a vel_msg that satisfies max change and max value constraints.
					
					vel_msg = turtlebot_twist(v,omega)

					# Update current v and omega
					self.v = v
					self.omega = omega

					self.vel_pub.publish(vel_msg)
					# self.get_logger().info('{}'.format(vel_msg))
				else:
					self.vel_pub.publish(stop_twist())

					# Update current v and omega
					self.v = 0.0
					self.omega = 0.0

		else:
			self.vel_pub.publish(stop_twist())
			self.motion_reset()

	# def action_projection(self, linear_velocity):
	# 	rotational_velocity = 0.5
	# 	if self.collsion_avoidance_neighbors is not None:
	# 		for robot_namespace in self.collsion_avoidance_neighbors:
	# 			my_loc = self.get_my_loc()
	# 			my_yaw = self.get_my_yaw()
	# 			neighbor_loc = self.robot_listeners[robot_namespace].get_latest_loc()
	# 			neighbor_yaw = self.robot_listeners[robot_namespace].get_latest_yaw()
	# 			neighbor_cmd_vel = self.robot_listeners[robot_namespace].get_latest_cmd_vel()
	# 			(neighbor_linear, neighbor_angular) = neighbor_cmd_vel
	# 			dist_direction_for_me = math.atan2(
	# 					neighbor_loc[1] - my_loc[1],
	# 					neighbor_loc[0] - my_loc[0])
	# 			distance = math.sqrt(
	# 					(neighbor_loc[1] - my_loc[1])**2 +
	# 					(neighbor_loc[0] - my_loc[0])**2)
	# 			dist_direction_for_neighbor = dist_direction_for_me + math.pi
	# 			dot_d = linear_velocity * math.cos(my_yaw - dist_direction_for_me)
	# 			if dot_d <= -1 * self.alpha_for_collision_avoidance * distance:
	# 				return np.sign()
	# 			else:
	# 				return 0.0
				





def main(args=sys.argv):
	rclpy.init(args=args)
	args_without_ros = rclpy.utilities.remove_ros_args(args)

	print(args_without_ros)
	arguments = len(args_without_ros) - 1
	position = 1


	# Get the robot name passed in by the user
	robot_namespace='MobileSensor1'
	if arguments >= position:
		robot_namespace = args_without_ros[position]
	
	if arguments >= position+1:
		pose_type_string = args_without_ros[position+1]
	else:
		pose_type_string = prompt_pose_type_string()
	
	if arguments >= position+2:
		neighborhood = args_without_ros[position+2].split(',')
	else:
		neighborhood = ['MobileSensor{}'.format(n) for n in range(1,5)]
		# neighborhood = set(['MobileSensor1'])
	
	if arguments >= position+3:
		init_target_position = [float(loc) for loc in args_without_ros[position+3].split(',')]
	else:
		init_target_position = [0.0, 1.0 * float(robot_namespace[-1])]
	
	if arguments >= position+4:
		data_dir = args_without_ros[position + 4]
	else:
		data_dir = None
	
	qhat_0 = (np.random.rand(2)-0.5)*0.5+np.array([1.5,2])

	x_max = 0
	x_min = -3
	y_max = 4
	y_min = 0
	
	# estimator = ConsensusEKF(qhat_0,R_mag=10,Q_mag = 10,C_gain=0.1,\
	# 	       # Dimensions about the lab, fixed.
	#             x_max = x_max,
	#             x_min = x_min,
	#             y_max = y_max,
	#             y_min = y_min)

	de = distributed_seeking(robot_namespace,pose_type_string, init_target_position, neighborhood_namespaces = neighborhood,
			  				data_dir=data_dir,
							xlims=[x_min-1,x_max+1],
							ylims=[y_min-1,y_max+1])
	
	de.get_logger().info(str(args_without_ros))
	try:
		print('Distributed Seeking Node Up')
		rclpy.spin(de)
	except KeyboardInterrupt:
		print("Keyboard Interrupt. Shutting Down...")
		for _ in range(30):# Publish consecutive stop twist for 3 seconds to ensure the robot steps.
			de.vel_pub.publish(stop_twist())
			time.sleep(0.1)
	finally:
		de.destroy_node()
		print('Distributed Seeking Node Down')
		rclpy.shutdown()

if __name__ == '__main__':
	main()