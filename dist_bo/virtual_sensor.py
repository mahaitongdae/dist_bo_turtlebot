import sys
import os
tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))


import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Float32MultiArray
from rclpy.qos import QoSProfile

import numpy as np


from ros2_utils.pose import prompt_pose_type_string
from ros2_utils.misc import get_sensor_names, get_source_names
from ros2_utils.robot_listener import robot_listener

class virtual_sensor(object):
	"""
	Remember to use virtual_sensor.py (not just virtual_sensor) to refer to this package.

	The virtual sensor used in simulations. 
	"""
	def __init__(self,C1,C0,b,k,noise_std):
		
		self.C1=C1
		self.C0=C0
		self.b=b
		self.k=k
		self.noise_std= noise_std

	def measurement(self,source_locs,sensor_locs):
		"""
		The measurement model: y = k(r-C_1)^b+C_0
		
		Output: y satisfying len(y)=len(sensor_locs)
		"""

		# Use the outter product technique to get the pairwise displacement between sources and sensors
		q2p=source_locs[:,np.newaxis]-sensor_locs

		# calculate the pairwise distance
		d=np.linalg.norm(q2p,axis=-1)


		y = (self.k*(d-self.C1)**self.b)+self.C0

		# Sum over the influence of all sources to get the total measurement for each sensor.
		y=np.sum(y,axis = 0)


		# Add noise
		y += np.random.randn(*y.shape)*self.noise_std

		# Avoid pathological values in location estimation.
		y[y<=0]=1e-7

		return y

class virtual_sensor_node(Node):
	def __init__(self,pose_type_string,sensor_namespaces=None,source_namespaces=None):
		super().__init__('virtual_sensor')

		self.pose_type_string = pose_type_string

		self.sensor_listeners = {}
		self.source_listeners = {}

		sleep_time = 0.5
		self.timer = self.create_timer(sleep_time,self.timer_callback)

		C1=-0.3
		C0=0
		b=-2
		k=1
		noise_std=0.01

		self.vs = virtual_sensor(C1,C0,b,k, noise_std)

		self.source_loc = ''
		self.sensor_locs = ''

		self.pubs = {}
		self.subs = {}

		sensor_namespaces = get_sensor_names(self) if sensor_namespaces is None else sensor_namespaces
		source_namespaces = get_source_names(self) if source_namespaces is None else source_namespaces

		self.subscriber_init(sensor_namespaces,source_namespaces)
		self.publisher_init(sensor_namespaces)

		self.get_logger().info('Sensor Init Done '+pose_type_string)

		self.coefs = {'C1':C1,'C0':C0,'k':k,'b':b}

	

	def subscriber_init(self,sensor_namespaces,source_namespaces):
		self.sensor_listeners = {namespace:robot_listener(self,namespace,self.pose_type_string) for namespace in sensor_namespaces}
		self.source_listeners = {namespace:robot_listener(self,namespace,self.pose_type_string) for namespace in source_namespaces}

	def publisher_init(self,robot_namespaces):
		qos = QoSProfile(depth=10)

		self.pubs = {namespace:self.create_publisher(Float32MultiArray,'/{}/sensor_readings'.format(namespace),qos) for namespace in robot_namespaces}
	


	def timer_callback(self):
		
		
		src = []
		for source_name,sls in self.source_listeners.items():
			if not sls.get_latest_loc() is None:
				src.append(sls.get_latest_loc())

		src = np.array(src).reshape(-1,2)

		self.get_logger().info('Timer Callback, ')
		for sensor_name, pub in self.pubs.items():
			
			# src = self.source_loc
			# sen = self.sensor_locs[sensor_name]

			ls = self.sensor_listeners[sensor_name]
			sen = ls.get_latest_loc()
			if not sen is None:
				msg = Float32MultiArray()

				msg.data = list(self.vs.measurement(src,sen))
				self.pubs[sensor_name].publish(msg)

def main(args = sys.argv):

	rclpy.init(args=args)
	args_without_ros = rclpy.utilities.remove_ros_args(args)

	arguments = len(args_without_ros) - 1
	position = 1

	# Get the pose type passed in by the user
	if arguments>=position:
		pose_type_string = sys.argv[position]
	else:
		pose_type_string = prompt_pose_type_string()

	sensors = None
	source = None

	if arguments>=position+1:
		sensors = sys.argv[position+1].split(',')
	if arguments>=position+2:
		source = sys.argv[position+2].split(',')
	
	tk = virtual_sensor_node(pose_type_string,sensors,source)	

	tk.get_logger().info(str(args_without_ros))
	try:
		print('Virtual Sensor Up')
		rclpy.spin(tk)
	except KeyboardInterrupt:
		print("Keyboard Interrupt. Shutting Down...")
	finally:
		tk.destroy_node()
		tk.get_logger().info('Virtual Sensor Down')
		rclpy.shutdown()

if __name__ == '__main__':
	main()