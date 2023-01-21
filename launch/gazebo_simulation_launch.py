import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

def spawn_object_exec(name,filepath,x,y,Yaw):
    return ExecuteProcess(
            cmd = "ros2 run fim_track_2 spawn_entity\
             -file {} -entity {} -robot_namespace {} -x {} -y {} -Y {}"\
            .format(filepath,name,name,x,y,Yaw).split(),
            output='screen')

def generate_launch_description():

    # source_names = ["Source{}".format(i) for i in range(1)]
    # # Initial position and orientation of moving sources
    # # t_x =[6]
    # # t_y = [6]
    # t_x =[3]
    # t_y = [0]
    # t_Yaw = [0]

    sensor_names = ['MobileSensor{}'.format(i + 1) for i in range(2)]

    # Initial position and orientation of sensors
    x = [0., 1.]
    y=[0., 1.]
    Yaw = [0,0,0,0]

    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = 'empty.model'
    
    moving_source_file = 'source_turtlebot.sdf'
    mobile_sensor_file = 'mobile_sensor.sdf'

    world = os.path.join(get_package_share_directory('fim_track_2'), 'worlds', world_file_name)
    moving_source_path = os.path.join(get_package_share_directory('fim_track_2'), 'models', moving_source_file)
    mobile_sensor_path = os.path.join(get_package_share_directory('fim_track_2'), 'models', mobile_sensor_file)

    # launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')

    execs = [
            ExecuteProcess(
            cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'],
            output='screen'),

            ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/gazebo', 'use_sim_time', use_sim_time],
            output='screen'),
            ]

    # # Append the source spawning commands to the execution list.
    # for i,name in enumerate(source_names):
    #     execs.append(spawn_object_exec(name,moving_source_path,t_x[i],t_y[i],t_Yaw[i]))

    # Append the sensor spawning commands to the execution list.
    for i,name in enumerate(sensor_names):
        execs.append(spawn_object_exec(name,mobile_sensor_path,x[i],y[i],Yaw[i]))

      
    # execs = []
    # execs.append(Node(package = 'fim_track_2',
    #             executable = 'virtual_sensor',
    #             arguments = ['Odom',','.join(sensor_names),','.join(source_names)]
    #             ))
    execs.extend([Node(package='fim_track_2',executable='virtual_coef',arguments=[name]) for name in sensor_names])

  

    return LaunchDescription(execs)
