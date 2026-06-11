import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('wheel_leg_description')

    urdf_file = os.path.join(
        pkg_share,
        'urdf',
        'wheel_leg_robot_gazebo_locked.urdf'
    )

    world_file = os.path.join(
        pkg_share,
        'worlds',
        'wheel_leg_ground.world'
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'pause': 'false',
            'verbose': 'true'
        }.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'wheel_leg_robot',
            '-x', '0',
            '-y', '0',
            '-z', '0.25'
        ],
        output='screen'
    )

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        gazebo,
        robot_state_publisher,
        TimerAction(period=3.0, actions=[spawn_robot]),
    ])
