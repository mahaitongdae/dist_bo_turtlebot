import time

import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Bool



def get_key(settings):
	tty.setraw(sys.stdin.fileno())
	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
	if rlist:
		key = sys.stdin.read(1)
	else:
		key = ''

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key



class MOVE(Node):
	def __init__(self):
		super().__init__('MOVE',namespace='MISSION_CONTROL')
		
		qos = QoSProfile(depth=10)
		self.pub_ = self.create_publisher(Bool,'MOVE',qos)

		self.settings = termios.tcgetattr(sys.stdin)
			
	def spin(self):
		self.stop()
		print('Press m to start moving, s to stop.')		
		while(1):
			key = get_key(self.settings)
			if key=='s':
				self.stop()
			elif key=='m':
				self.move()
			else:
				if (key == '\x03'):
					self.stop()
					break
	
	def move(self):
		msg = Bool()
		msg.data = True
		print('Moving.')
		self.pub_.publish(msg)

	def stop(self):
		msg = Bool()
		msg.data = False
		print('Stopping.')
		self.pub_.publish(msg)
		

def main():
	rclpy.init()
	move = MOVE()

	try:
		move.spin()
	except KeyboardInterrupt:
		print("Keyboard Interrupt. Stopping robots...")
	finally:
		print("Keyboard Interrupt. Shutting Down...")
		for _ in range(30):# Publish stop twist for 3 seconds to ensure the robot steps.
			move.stop()
			time.sleep(0.1)
		move.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()