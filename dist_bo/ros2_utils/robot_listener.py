from functools import partial

import rclpy
from rclpy.qos import QoSProfile
from rcl_interfaces.srv import GetParameters

from std_msgs.msg import Float32MultiArray,Float32

from ros2_utils.pose import get_pose_type_and_topic,toxy,toyaw

from collections import deque


class robot_listener:
	''' Robot location and light_reading listener+data container.'''
	def __init__(self,controller_node,robot_namespace,pose_type_string="",max_record_len=10):
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
		"""
		self.robot_name=robot_namespace

		controller_node.get_logger().info('initializing {} listener'.format(robot_namespace))
		
		
		self.pose_type,self.rpose_topic=get_pose_type_and_topic(pose_type_string,robot_namespace)
		
		self.light_topic="/{}/sensor_readings".format(robot_namespace)
		self.coef_topic="/{}/sensor_coefs".format(robot_namespace)
		self.robot_pose_stack = deque(maxlen=10)
		self.light_readings_stack = deque(maxlen=10)

		
		qos = QoSProfile(depth=10)

		controller_node.create_subscription(self.pose_type, self.rpose_topic,self.robot_pose_callback_,qos)
		controller_node.create_subscription(Float32MultiArray,self.light_topic, self.light_callback_,qos)
		controller_node.create_subscription(Float32MultiArray,self.coef_topic,self.coef_callback_,qos)
	
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

	def get_coefs(self):
		return self.coefs


	def robot_pose_callback_(self,data):
		self.robot_pose_stack.append(data)

	def light_callback_(self,data):
		self.light_readings_stack.append(data.data)

	def coef_callback_(self,data):
		d = data.data
		self.coefs['k'] = d[0]
		self.coefs['b'] = d[1]
		self.coefs['C0'] = d[2]
		self.coefs['C1'] = d[3] 
