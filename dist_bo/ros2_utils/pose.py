import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose,Twist,PoseStamped
from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry

BURGER_MAX_LIN_VEL = 0.22 * 0.8
BURGER_MAX_ANG_VEL = 2.84 * 0.8


LIN_VEL_STEP_SIZE = 0.1
ANG_VEL_STEP_SIZE = 1

def prompt_pose_type_string():
    platform_2_pose_types=dict()
    platform_2_pose_types['s']="turtlesimPose"
    platform_2_pose_types['g']='Odom'
    platform_2_pose_types['t']='Pose'
    platform_2_pose_types['o']='optitrack'

    platform=input("Please indicate the platform of your experiment.\n s => turtlesim\n g => Gazebo\n s => SLAM \n o => Optitrack:")
    return platform_2_pose_types[platform]

def get_pose_type_and_topic(pose_type_string,robot_namespace):

    """
        pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
    """
    
    if pose_type_string=='turtlesimPose':
        pose_type_string=tPose
        rpose_topic="/{}/pose".format(robot_namespace)
    elif pose_type_string=='Pose':
        pose_type_string=Pose
        rpose_topic="/{}/pose".format(robot_namespace)
    elif pose_type_string=='Odom':
        pose_type_string=Odometry
        rpose_topic="/{}/odom".format(robot_namespace)
    elif pose_type_string=='optitrack':
        pose_type_string=PoseStamped
        rpose_topic="/vrpn_client_node/{}/pose".format(robot_namespace)
    else:
        print('Unknown pose:[',pose_type_string,']')
        rpose_topic=''
    return pose_type_string,rpose_topic

def toyaw(pose):
    if type(pose) is tPose:
        return tPose2yaw(pose)
    elif type(pose) is Odometry:
        return yaw_from_odom(pose)
    elif type(pose) is PoseStamped:
        return posestmp2yaw(pose)
    else:
        print('Pose to yaw conversion is not yet handled for {}'.format(type(pose)))
        return None

def toxy(pose):
    if type(pose) is tPose:
        return tPose2xy(pose)
    elif type(pose) is Odometry:
        return xy_from_odom(pose)
    elif type(pose) is Pose:
        return pose2xz(pose)
    elif type(pose) is PoseStamped:
        return posestmp2xy(pose)
    else:
        print('Pose to xy conversion is not yet handled for {}'.format(type(pose)))
        assert(False)
        return None
"""
pose is the Standard pose type as defined in geometry_msgs
"""
def pose2xz(pose):
    return np.array([pose.position.x,pose.position.z])

"""
tPose stands for the ROS data type turtlesim/Pose
"""
def tPose2xy(data):
        return np.array([data.x,data.y])
def tPose2yaw(data):
        return data.theta
    

"""
The following are the location/yaw converters from Odometry.
"""

def quaternion2yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def yaw_from_odom(odom):
    return quaternion2yaw(odom.pose.pose.orientation)
def xy_from_odom(odom):
    return np.array([odom.pose.pose.position.x,odom.pose.pose.position.y])
"""
The following converts PoseStampe data to x,z and yaw
"""
def posestmp2xy(pose):
    return np.array([pose.pose.position.x,pose.pose.position.y])
def posestmp2yaw(pose):
    return quaternion2yaw(pose.pose.orientation)




def bounded_change_update(target_v,target_omega,curr_v,curr_omega):

    def make_simple_profile(output, input, slop):
        if input > output:
            output = min( input, output + slop )
        elif input < output:
            output = max( input, output - slop )
        else:
            output = input

        return output

    v = make_simple_profile(curr_v,target_v,LIN_VEL_STEP_SIZE/2.0)
    omega = make_simple_profile(curr_omega,target_omega,ANG_VEL_STEP_SIZE/2.0)

    return [v,omega]

def turtlebot_twist(v,omega):
    def constrain(input, low, high):
        if input < low:
            input = low
        elif input > high:
            input = high
        else:
            input = input
        return input

    vel_msg=Twist()
    # Linear velocity in the x-axis.
    
    vel_msg.linear.x = constrain(v,-BURGER_MAX_LIN_VEL,BURGER_MAX_LIN_VEL)
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0.0

    # Angular velocity in the z-axis.
    vel_msg.angular.x = 0.0
    vel_msg.angular.y = 0.0
    vel_msg.angular.z = constrain(omega,-BURGER_MAX_ANG_VEL,BURGER_MAX_ANG_VEL)
    return vel_msg

def stop_twist():
    twist = Twist()
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0

    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0
    return twist