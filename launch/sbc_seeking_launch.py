import socket

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node

import networkx as nx

def neighorhoods(G, mobile_sensors):

	G=nx.relabel_nodes(G,{i:sensor for (i,sensor) in enumerate(mobile_sensors)})
	return {sensor:list(dict(G[sensor]).keys())+[sensor] for sensor in mobile_sensors}

def generate_launch_description():
	# mobile_sensors = ['MobileSensor{}'.format(i) for i in range(1,2)]
	mobile_sensors = ['MobileSensor6']

	G = nx.circulant_graph(len(mobile_sensors), [0,1])

	G.remove_edges_from(nx.selfloop_edges(G))

	nb = neighorhoods(G,mobile_sensors)

	my_name = socket.gethostname()


# def generate_launch_description():
	# return LaunchDescription([
	# 	Node(package = 'dist_bo',
	# 	executable = 'distributed_seeking',
	# 	arguments = [my_name,'optitrack',','.join(nb[my_name])]
	# 	) ])
	execs = []

	execs.extend([Node(package = 'dist_bo',
			executable = 'centralized_decision',
			arguments = ['optitrack', ','.join(mobile_sensors)])])

	# execs.extend([Node(package = 'dist_bo',
	# 		executable = 'visualize',
	# 		arguments = [])])

	init_location = ['-1.0,-1.0', '-2.0,-1.0', '2.0,-2.0', '-2.0,-2.0']
	
	execs.extend([Node(package = 'dist_bo',
			executable = 'distributed_exploration',
			arguments = [name,'optitrack',','.join(sorted(nb[name])), init_location[i]]
			) for i, name in enumerate(mobile_sensors)])
	
	return LaunchDescription(execs)
