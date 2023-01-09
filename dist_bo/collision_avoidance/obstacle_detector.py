import os
import sys

tools_root = os.path.join(".."+os.path.dirname(__file__))
print(tools_root)
sys.path.insert(0, os.path.abspath(tools_root))


import numpy as np

from rcl_interfaces.srv import GetParameters

from ros2_utils.robot_listener import robot_listener
from ros2_utils.param_service_client import param_service_client
from ros2_utils.misc import get_sensor_names,get_source_names
from collision_avoidance import regions

# The radius of a Turtlebot Burger. Useful in collision avoidance.
BURGER_RADIUS = 0.110

class obstacle_detector:
	def __init__(self,mc_node):
		self.obs_names = get_sensor_names(mc_node)
		self.ol = [robot_listener(mc_node,name,mc_node.pose_type_string) for name in self.obs_names if (not name == mc_node.robot_namespace) and (not name=='Source0')]

	def get_free_spaces(self):
		
		SAFE_RADIUS = 3*BURGER_RADIUS
		# SAFE_RADIUS = 5*BURGER_RADIUS
		# SAFE_RADIUS = 6*BURGER_RADIUS

		obs = [(l.get_latest_loc(),SAFE_RADIUS) for l in self.ol if not l.get_latest_loc() is None]

		return [regions.CircleExterior(origin,radius) for (origin,radius) in obs]

class boundary_detector:
	def __init__(self,controller_node,xlims,ylims):
		self.xlims = xlims
		self.ylims = ylims
		# self.xlims = (-1e5,1e5)
		# self.ylims = (-1e5,1e5)
		
		# # Get boundary services.
		# self.param_names = ['xlims','ylims']
		# self.param_service = '/MISSION_CONTROL/boundary/get_parameters'

		# self.boundary_client = param_service_client(controller_node,self.param_names,self.param_service)


	def get_free_spaces(self):
		# result = self.boundary_client.get_params()
		# if len(result)>0:
		# 		[self.xlims,self.ylims] = result

		return [regions.Rect2D(self.xlims,self.ylims)]

class source_contact_detector:
	def __init__(self,mc_node):
		# self.src_names = get_source_names(mc_node)
		self.src_names = ['Source0']
		self.sl = [robot_listener(mc_node,name,mc_node.pose_type_string) for name in self.src_names]
		self.mc_node = mc_node

	def contact(self):

		sensor_loc = self.mc_node.get_my_loc()
		
		if sensor_loc is None:
			return False
		else:
			src_loc = [l.get_latest_loc() for l in self.sl if not l.get_latest_loc() is None]
			if len(src_loc)==0:
				return False

			src_loc = np.array(src_loc).reshape(-1,len(sensor_loc))
			
			# self.mc_node.get_logger().info(str(src_loc)+','+str(sensor_loc)+str(self.src_names))
			return np.any(np.linalg.norm(src_loc-sensor_loc)<= 5*BURGER_RADIUS)

	def get_source_loc(self):
		if self.contact():
			src_loc = [l.get_latest_loc() for l in self.sl if not l.get_latest_loc() is None]
			if len(src_loc)==0:
				return None

			sensor_loc = self.mc_node.get_my_loc()

			src_loc = np.array(src_loc).reshape(-1,len(sensor_loc))
			return src_loc
		else:
			return None

