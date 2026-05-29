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
        full_auto_path = get_package_share_directory('full_auto_rover')
        peripherals_path = get_package_share_directory('peripherals')
        controller_path = get_package_share_directory('controller')
        kinematics_path = get_package_share_directory('kinematics')
    else:
        full_auto_path = '/home/ubuntu/ros2_ws/src/full_Auto_Rover'
        peripherals_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        kinematics_path = '/home/ubuntu/ros2_ws/src/driver/kinematics'

    event_config = os.path.join(full_auto_path, 'config/event_rules.yaml')
    tracker_config = os.path.join(full_auto_path, 'config/camera_tracker.yaml')
    control_config = os.path.join(full_auto_path, 'config/robot_control.yaml')

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

    start_arm_controller = LaunchConfiguration('start_arm_controller').perform(context).lower() == 'true'

    nodes.extend([
        Node(
            package='full_auto_rover',
            executable='yolo_object_detector',
            output='screen',
            parameters=[event_config],
        ),
        Node(
            package='full_auto_rover',
            executable='danger_event_detector',
            output='screen',
            parameters=[event_config],
        ),
        Node(
            package='full_auto_rover',
            executable='event_camera_tracker',
            output='screen',
            parameters=[tracker_config],
        ),
        Node(
            package='full_auto_rover',
            executable='snapshot_sender',
            output='screen',
            parameters=[tracker_config],
        ),
        Node(
            package='full_auto_rover',
            executable='motion_controller',
            output='screen',
            parameters=[control_config],
        ),
        Node(
            package='full_auto_rover',
            executable='mission_coordinator',
            output='screen',
            parameters=[control_config],
        ),
    ])
    if start_arm_controller:
        nodes.append(Node(
            package='full_auto_rover',
            executable='arm_controller',
            output='screen',
            parameters=[control_config],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_controller', default_value='true'),
        DeclareLaunchArgument('start_kinematics', default_value='true'),
        DeclareLaunchArgument('start_arm_controller', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
