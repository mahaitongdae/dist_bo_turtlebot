
import re

def get_sensor_names(curr_node):
	sensor_names = set()
	for (topic,_type) in curr_node.get_topic_names_and_types():
		topic_split = re.split('/',topic)
		if ('pose' in topic_split) or ('odom' in topic_split):
			# pose_type_string = topic[1]
			name = re.search('/MobileSensor.*/',topic)
			if not name is None:
				sensor_names.add(name.group()[1:-1])
	return list(sensor_names)

def get_source_names(curr_node):
	sensor_names = set()
	for (topic,_type) in curr_node.get_topic_names_and_types():
		topic_split = re.split('/',topic)
		if ('pose' in topic_split) or ('odom' in topic_split):
			# pose_type_string = topic[1]
			name = re.search('/Source.*/',topic)
			if not name is None:
				sensor_names.add(name.group()[1:-1])
	return list(sensor_names)
