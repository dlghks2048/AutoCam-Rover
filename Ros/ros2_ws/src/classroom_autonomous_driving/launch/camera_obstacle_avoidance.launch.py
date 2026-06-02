import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def package_path(package_name, source_path):
    if os.environ.get('need_compile', 'False') == 'True':
        return get_package_share_directory(package_name)
    return source_path


def launch_setup(context):
    os.environ.setdefault('need_compile', 'True')

    classroom_path = package_path(
        'classroom_autonomous_driving',
        '/home/ubuntu/ros2_ws/src/classroom_autonomous_driving',
    )
    peripherals_path = package_path('peripherals', '/home/ubuntu/ros2_ws/src/peripherals')
    controller_path = package_path('controller', '/home/ubuntu/ros2_ws/src/driver/controller')

    nodes = []
    if LaunchConfiguration('start_controller').perform(context).lower() == 'true':
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(controller_path, 'launch/controller.launch.py'))))

    if LaunchConfiguration('start_camera').perform(context).lower() == 'true':
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(peripherals_path, 'launch/depth_camera.launch.py'))))

    nodes.append(Node(
        package='classroom_autonomous_driving',
        executable='camera_obstacle_avoidance',
        output='screen',
        parameters=[os.path.join(classroom_path, 'config/camera_obstacle_avoidance.yaml')],
    ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_controller', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
