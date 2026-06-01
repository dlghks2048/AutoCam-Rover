import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':
        final_path = get_package_share_directory('final_compilation')
        peripherals_path = get_package_share_directory('peripherals')
        controller_path = get_package_share_directory('controller')
        kinematics_path = get_package_share_directory('kinematics')
    else:
        final_path = '/home/ubuntu/ros2_ws/src/final_compilation'
        peripherals_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        kinematics_path = '/home/ubuntu/ros2_ws/src/driver/kinematics'

    config = os.path.join(final_path, 'config/integrated_auto_rover.yaml')
    nodes = []

    if LaunchConfiguration('start_controller').perform(context).lower() == 'true':
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(controller_path, 'launch/controller.launch.py'))))

    if LaunchConfiguration('start_kinematics').perform(context).lower() == 'true':
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(kinematics_path, 'launch/kinematics_node.launch.py'))))

    if LaunchConfiguration('start_camera').perform(context).lower() == 'true':
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(peripherals_path, 'launch/depth_camera.launch.py'))))

    nodes.append(Node(
        package='final_compilation',
        executable='integrated_auto_rover_node',
        output='screen',
        parameters=[config],
    ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_controller', default_value='true'),
        DeclareLaunchArgument('start_kinematics', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
