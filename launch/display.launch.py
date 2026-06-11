from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('wheel_leg_description')

    urdf_file = PathJoinSubstitution([
        pkg_share,
        'urdf',
        'wheel_leg_robot.urdf'
    ])

    rviz_config = PathJoinSubstitution([
        pkg_share,
        'rviz',
        'wheel_leg_robot.rviz'
    ])

    robot_description = ParameterValue(
        Command(['cat ', urdf_file]),
        value_type=str
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),

        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        )
    ])
