#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert

import math
import time
FEEDBACK_P = 5.

from geometry_msgs.msg import Twist

def simple_truncate(control, upper, lower):
    return min(max(control, lower), upper)


class Turtlebot3Path(object):
    def __init__(self) -> None:
        self.last_dist = 100.0 # a large value that must be larger than the first distance
    
    def turn(self, angle, angular_velocity, step):
        twist = Twist()
        if angle <= -math.pi:
            angle = angle + 2 * math.pi
        fbk_anglar_vel = simple_truncate(FEEDBACK_P*math.fabs(angle), angular_velocity, 0.1*angular_velocity)
        if math.fabs(angle) > 0.01:  # 0.01 is small enough value
            if angle >= math.pi:
                twist.angular.z = -fbk_anglar_vel
            elif math.pi > angle and angle >= 0:
                twist.angular.z = fbk_anglar_vel
            elif 0 > angle and angle >= -math.pi:
                twist.angular.z = -fbk_anglar_vel
            elif angle > -math.pi:
                twist.angular.z = fbk_anglar_vel
        else:
            time.sleep(0.5)
            step += 1

        return twist, step

    def go_straight(self, distance, linear_velocity, step):
        twist = Twist()
        if distance > 0.03:  # 0.01 is small enough value
            twist.linear.x = simple_truncate(FEEDBACK_P * distance, linear_velocity, 0.5 * linear_velocity)
            if distance > self.last_dist:
                step -= 1
                twist.linear.x = 0.
                time.sleep(0.5)
        else:
            step += 1
        self.last_dist = distance
        return twist, step