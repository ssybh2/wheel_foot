import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('wheel_leg_description')

    sdf_file = os.path.join(
        pkg_share,
        'sdf',
        'wheel_leg_robot_closed_loop_control.sdf'
    )

    world_file = os.path.join(
        pkg_share,
        'worlds',
        'wheel_leg_ground.world'
    )

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

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', sdf_file,
            '-entity', 'wheel_leg_robot',
            '-x', '0',
            '-y', '0',
            '-z', '0.10'
        ],
        output='screen'
    )
    kinematics_node = Node(
        package='wheel_leg_description',
        executable='five_bar_kinematics_node.py',
        name='five_bar_kinematics',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ],
    )
    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        gazebo,
        TimerAction(period=4.0, actions=[spawn_robot]),
        TimerAction(period=5.0, actions=[kinematics_node]),
    ])
