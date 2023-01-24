from setuptools import setup
from glob import glob
import os

package_name = 'dist_bo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*launch.py')),
        ('share/' + package_name+'/param', glob('param/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.model')),
        ('share/' + package_name + '/models', glob('models/*.sdf')),
        ('share/'+package_name+'/models/mobile_sensor/meshes/', glob('models/mobile_sensor/meshes/*')),
              ('share/'+package_name+'/models/mobile_sensor/', glob('models/mobile_sensor/*.*')),
   ('share/'+package_name+'/models/source_turtlebot/meshes/', glob('models/source_turtlebot/meshes/*')),
              ('share/'+package_name+'/models/source_turtlebot/', glob('models/source_turtlebot/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mht',
    maintainer_email='haitongma@g.harvard.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'distributed_exploration = dist_bo.distributed_exploration:main',
            'visualize = dist_bo.visualization:main',
            'virtual_source = dist_bo.virtual_source_func:main',
            'virtual_sensor = dist_bo.virtual_sensor:main',
            'centralized_decision = dist_bo.centralized_decision:main',
            'client = dist_bo.client_member_function:main',
            'spawn_entity = dist_bo.spawn_entity:main',
        ],
    },
)
