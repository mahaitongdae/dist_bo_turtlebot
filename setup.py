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
        ],
    },
)
