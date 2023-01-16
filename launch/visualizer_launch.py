from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

	execs = []
	execs.extend([Node(package = 'dist_bo',
			executable = 'visualize',
			arguments = ['']
			) ])

	return LaunchDescription(execs)