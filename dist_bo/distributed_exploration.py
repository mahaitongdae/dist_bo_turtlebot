# The big distributed seeking node. This reduces the number of topic subscription needed.
import os
import sys
import traceback
import time

import numpy as np

from functools import partial
from collections import deque

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray,Bool,String
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node



tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, '/home/mht/turtlebot3_ws/src/dist_bo/dist_bo') #todo:temp solution

# General dependencies
from ros2_utils.robot_listener import robot_listener
from ros2_utils.pose import prompt_pose_type_string
from ros2_utils.misc import get_sensor_names


# Estimation dependencies
from estimation.ConsensusEKF import ConsensusEKF 
from util_func import joint_meas_func, analytic_dhdz, top_n_mean

# Waypoint planning dependencies
from motion_control import WaypointPlanning

from motion_control.WaypointTracking import BURGER_MAX_LIN_VEL

from util_func import analytic_dLdp, joint_F_single
from consensus import consensus_handler

# Motion control dependencies
from ros2_utils.pose import bounded_change_update, turtlebot_twist, stop_twist

from motion_control.WaypointTracking import LQR_for_motion_mimicry

from collision_avoidance.obstacle_detector import obstacle_detector, source_contact_detector, boundary_detector
from collision_avoidance.regions import RegionsIntersection


COEF_NAMES = ['C1','C0','k','b']

def get_control_action(waypoints,curr_x):
	if len(waypoints)==0:
		return []
		
	planning_dt = 0.1

	Q = np.array([[10,0,0],[0,10,0],[0,0,1]])
	R = np.array([[10,0],[0,1]])

	uhat,_,_ = LQR_for_motion_mimicry(waypoints,planning_dt,curr_x,Q=Q,R=R)

	return uhat

class distributed_seeking(Node):
	def __init__(self, robot_namespace, pose_type_string, neighborhood_namespaces=None, xlims=[-np.inf,np.inf], ylims = [-np.inf,np.inf]):
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



		qos = QoSProfile(depth=10)

		"""
		Timer initialization
		"""

		# self.est_sleep_time = 0.1

		# self.est_sleep_time = 2
		
		# self.estimation_timer = self.create_timer(self.est_sleep_time,self.est_callback)

		self.waypoint_sleep_time = 0.5

		self.waypoint_timer = self.create_timer(self.waypoint_sleep_time,self.simple_waypoint_callback)

		# self.FIM_consensus_sleep_time = 0.1
		
		# self.FIM_consensus_timer = self.create_timer(self.FIM_consensus_sleep_time,self.FIM_consensus_callback)		

		self.motion_sleep_time = 0.1
		
		# self.motion_sleep_time = 0.5

		self.motion_timer = self.create_timer(self.motion_sleep_time,self.simple_motion_callback)

		# self.visulize_sleep_tme = 0.1
		# self.visulize_timer = self.create_timer(self.visulize_sleep_tme, self.placeholder)


		# """ 
		# Estimation initializations 
		# """

		# self.z_hat_listeners = \
		# {namespace:
		# self.create_subscription(
		# 	Float32MultiArray,
		# 	'/{}/z_hat'.format(namespace),
		# 	partial(self.z_hat_callback,namespace = namespace),
		# 	qos)
		# 	for namespace in neighborhood_namespaces}

		# self.nb_zhats = {namespace:[] for namespace in neighborhood_namespaces}

		# self.z_hat_pub = self.create_publisher(Float32MultiArray,'z_hat',qos)
		# self.q_hat_pub = self.create_publisher(Float32MultiArray,'q_hat',qos)

		# self.estimator = estimator

		# self.q_hat = self.estimator.get_q()

		""" 
		Waypoint planning initializations 
		"""
			
		# Temporary hard-coded waypoints used in devel.	
		self.waypoints = np.array([[1.,-2.5],[2.,-1.5],[3,-0.5]])
		# self.waypoints = np.array([[3,-0.5]])

		self.F = 0

		# # FIM consensus handler
		# self.FIM = np.ones((2,2))*1e-4
		# self.cons = consensus_handler(self,robot_namespace,neighborhood_namespaces,self.FIM,topic_name = 'FIM',qos=qos)
		
		self.waypoint_pub = self.create_publisher(Float32MultiArray,'waypoints',qos)
	
		"""
		Motion control initializations
		"""

		self.MOVE = True
		self.move_sub = self.create_subscription(Bool,'/MISSION_CONTROL/MOVE',self.MOVE_CALLBACK, qos)
	
		self.vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

		self.control_actions = deque([])

		# Obstacles are expected to be circular ones, parametrized by (loc,radius)
		self.obstacle_detector = obstacle_detector(self)
		self.source_contact_detector = source_contact_detector(self)
		self.boundary_detector = boundary_detector(self,xlims,ylims)


		# If source is foumd by some robot, SOURCE_LOC will be set to not None.
		# self.check_source_found_sleep_time = 0.1

		# self.check_source_found_timer = self.create_timer(self.check_source_found_sleep_time,self.check_source_found_)

		# self.SOURCE_LOC_pub = self.create_publisher(Float32MultiArray, '/SOURCE_LOC', qos)
		# self.FIRST_FOUND_pub = self.create_publisher(String, '/FIRST_CONTACT_ROBOT', qos)

		# self.SOURCE_LOC_sub = self.create_subscription(Float32MultiArray,'/SOURCE_LOC',self.SOURCE_LOC_callback_,qos)
		# self.FIRST_FOUND_sub = self.create_subscription(String,'/FIRST_CONTACT_ROBOT',self.FIRST_FOUND_callback_,qos)


		self.FIRST_CONTACT_ROBOT = None
		self.SOURCE_LOC = None

		# current control actions
		self.v = 0.0
		self.omega = 0.0

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

	def consensus_weights(self,y,p):
		# assert(len(y)==len(p))
		# # Temporary hard-coded equally consensus weights. Making sure the consensus weights sum to one.
		# N_neighbor = len(y)
		# return np.ones(N_neighbor)/N_neighbor
		return None

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
			target_loc = np.array([1.,-2.5])
			self.waypoints = WaypointPlanning.straight_line(target_loc,my_loc\
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
		
		
	



def main(args=sys.argv):
	rclpy.init(args=args)
	args_without_ros = rclpy.utilities.remove_ros_args(args)

	print(args_without_ros)
	arguments = len(args_without_ros) - 1
	position = 1


	# Get the robot name passed in by the user
	robot_namespace='MobileSensor1'
	# if arguments >= position:
	# 	robot_namespace = args_without_ros[position]
	
	if arguments >= position+1:
		pose_type_string = args_without_ros[position+1]
	else:
		pose_type_string = prompt_pose_type_string()
	
	if arguments >= position+2:
		neighborhood = set(args_without_ros[position+2].split(','))
	else:
		# neighborhood = set(['MobileSensor{}'.format(n) for n in range(1,5)])
		neighborhood = set(['MobileSensor1'])
	
	
	qhat_0 = (np.random.rand(2)-0.5)*0.5+np.array([1.5,2])
	# qhat_0 = np.array([-1,-0.0])
	# qhat_0 = np.array([-1,0])
	# qhat_0 = np.array([4,-1])
	# qhat_0 = np.array([3,0])
	


	# estimator = ConsensusEKF(qhat_0)
	# estimator = ConsensusEKF(qhat_0,C_gain=0.1)

	x_max = 3
	x_min = 0
	y_max = 4
	y_min = 0
	
	estimator = ConsensusEKF(qhat_0,R_mag=10,Q_mag = 10,C_gain=0.1,\
		       # Dimensions about the lab, fixed.
	            x_max = x_max,
	            x_min = x_min,
	            y_max = y_max,
	            y_min = y_min)

	de = distributed_seeking(robot_namespace,pose_type_string, neighborhood_namespaces = neighborhood,
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