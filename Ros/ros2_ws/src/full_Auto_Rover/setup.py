import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'full_auto_rover'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Autonomous event detection and camera alignment pipeline for AutoCam Rover.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'danger_event_detector = full_auto_rover.danger_event_detector:main',
            'event_camera_tracker = full_auto_rover.event_camera_tracker:main',
            'snapshot_sender = full_auto_rover.snapshot_sender:main',
            'yolo_object_detector = full_auto_rover.yolo_object_detector:main',
            'motion_controller = full_auto_rover.motion_controller:main',
            'arm_controller = full_auto_rover.arm_controller:main',
            'mission_coordinator = full_auto_rover.mission_coordinator:main',
            'danger_scan_node = full_auto_rover.danger_scan_node:main',
        ],
    },
)
