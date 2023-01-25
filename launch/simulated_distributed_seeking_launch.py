from launch import LaunchDescription
from launch_ros.actions import Node

import networkx as nx

def neighorhoods(G, mobile_sensors):

	G=nx.relabel_nodes(G,{i:sensor for (i,sensor) in enumerate(mobile_sensors)})
	return {sensor:list(dict(G[sensor]).keys())+[sensor] for sensor in mobile_sensors}

def generate_launch_description():
	mobile_sensors = ['MobileSensor{}'.format(i+1) for i in range(3)] # 

	G = nx.circulant_graph(len(mobile_sensors), [0,1,2])

	G.remove_edges_from(nx.selfloop_edges(G))

	nb = neighorhoods(G,mobile_sensors)

	execs = []

	# execs.extend([Node(package = 'fim_track_2',
	# 			executable = 'distributed_estimation',
	# 			arguments = [name,'Odom',','.join(nb[name])]
	# 			) for name in mobile_sensors ])

	# execs.extend([Node(package = 'fim_track_2',
	# 			executable = 'waypoint_planning',
	# 			arguments = [name,'Odom',','.join(nb[name])]
	# 			) for name in mobile_sensors ])
	
	# execs.extend([Node(package = 'fim_track_2',
	# 			executable = 'single_robot_controller',
	# 			arguments = [name,'Odom']
	# 			) for name in mobile_sensors ])

	execs.extend([Node(package = 'dist_bo',
			executable = 'virtual_source',
			arguments = [])])

	execs.extend([Node(package = 'dist_bo',
			executable = 'centralized_decision',
			arguments = ['Odom', ','.join(mobile_sensors)])])

	# execs.extend([Node(package = 'dist_bo',
	# 		executable = 'visualize',
	# 		arguments = [])])

	init_location = ['2.0,2.0', '-2.0,2.0', '2.0,-2.0', '-2.0,-2.0']
	
	execs.extend([Node(package = 'dist_bo',
			executable = 'distributed_exploration',
			arguments = [name,'Odom',','.join(sorted(nb[name])), init_location[i]]
			) for i, name in enumerate(mobile_sensors)])
	
	
	return LaunchDescription(execs)