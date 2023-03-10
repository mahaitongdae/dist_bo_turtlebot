from functools import partial

import rclpy
from rclpy.qos import QoSProfile
from rcl_interfaces.srv import GetParameters

from std_msgs.msg import Float32MultiArray,Float32,Bool
from geometry_msgs.msg import Twist

from ros2_utils.pose import get_pose_type_and_topic,toxy,toyaw

from collections import deque


class robot_listener:
	''' Robot location and light_reading listener+data container.'''
	def __init__(self,controller_node,robot_namespace,pose_type_string="",max_record_len=10, central=False, visualizer=False):
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
		"""
		self.robot_name=robot_namespace

		self.is_central = central

		controller_node.get_logger().info('initializing {} listener'.format(robot_namespace))
		
		
		self.pose_type,self.rpose_topic=get_pose_type_and_topic(pose_type_string,robot_namespace)
		
		self.light_topic="/{}/sensor_readings".format(robot_namespace)
		self.reach_target_topic="/{}/target_reached".format(robot_namespace)
		self.observation_topic="/{}/observation".format(robot_namespace)
		self.new_query_topic="/{}/new_queries".format(robot_namespace)
		self.cmd_vel_topic="/{}/cmd_vel".format(robot_namespace)
		self.robot_pose_stack = deque(maxlen=10)
		self.observed_values_stack = deque(maxlen=10)
		self.reach_target_stack = deque(maxlen=10)
		self.light_readings_stack = deque(maxlen=10)
		self.new_queries_stack = deque(maxlen=10)
		self.cmd_vel_stack = deque(maxlen=10)
		

		
		qos = QoSProfile(depth=10)

		controller_node.create_subscription(self.pose_type, self.rpose_topic,self.robot_pose_callback_,qos)
		controller_node.create_subscription(Float32MultiArray, self.light_topic, self.light_callback_,qos)
		controller_node.create_subscription(Bool, self.reach_target_topic, self.reach_target_callback_,qos)
		
		controller_node.create_subscription(Float32MultiArray, self.new_query_topic, self.new_queries_callback,qos)
		controller_node.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback,qos)
		if central:
			controller_node.create_subscription(Float32, self.observation_topic, self.observed_value_callback,qos)
		if visualizer:
			self.look_ahead_topic="/{}/look_ahead".format(robot_namespace)
			self.look_ahead_stack = deque(maxlen=10)
			controller_node.create_subscription(Float32MultiArray, self.look_ahead_topic, self.look_ahead_callback,qos)
	
		self.coefs = {}


	def get_latest_loc(self):
		if len(self.robot_pose_stack)>0:
			return toxy(self.robot_pose_stack[-1])
		else:
			return None

	def get_latest_yaw(self):
		if len(self.robot_pose_stack)>0:
			return toyaw(self.robot_pose_stack[-1])
		else:
			return None

	def get_latest_readings(self):
		if len(self.light_readings_stack)>0:
			return self.light_readings_stack[-1]
		else:
			return None
		
	def get_observed_values(self):
		if self.is_central:
			if len(self.observed_values_stack) > 0:
				return self.observed_values_stack[-1]
			else:
				return None
		else:
			return None
	
	def get_new_queries(self):
		if len(self.new_queries_stack) > 0:
			return self.new_queries_stack[-1]
		else:
			return None
	
	def get_latest_cmd_vel(self):
		if len(self.cmd_vel_stack) > 0:
			return self.cmd_vel_stack[-1]
		else:
			return None
	
	def get_look_ahead_target(self):
		if len(self.look_ahead_stack) > 0:
			return self.look_ahead_stack[-1]
		else:
			return None

	def is_target_reached(self):
		if len(self.reach_target_stack)>0:
			return self.reach_target_stack[-1]
		else:
			return False

	def robot_pose_callback_(self,data):
		self.robot_pose_stack.append(data)

	def light_callback_(self,data):
		self.light_readings_stack.append(data.data)

	def reach_target_callback_(self, data):
		self.reach_target_stack.append(data.data)

	def observed_value_callback(self, data):
		self.observed_values_stack.append(data.data)
	
	def new_queries_callback(self, data):
		self.new_queries_stack.append(data.data)

	def reset_obs_target_stacks(self):
		self.observed_values_stack = deque(maxlen=10)
		self.reach_target_stack = deque(maxlen=10)

	def cmd_vel_callback(self, data):
		self.cmd_vel_stack.append((data.linear.x, data.angular.z))

	def look_ahead_callback(self, data):
		self.look_ahead_stack.append(data.data)

	

	