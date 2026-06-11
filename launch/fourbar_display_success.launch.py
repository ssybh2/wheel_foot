import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('wheel_leg_description')

    urdf_file = os.path.join(
        pkg_share,
        'urdf',
        'wheel_leg_robot.urdf'
    )

    rviz_config = os.path.join(
        pkg_share,
        'rviz',
        'wheel_leg_robot.rviz'
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description}
        ]
    )

    fourbar_joint_publisher = Node(
        package='wheel_leg_description',
        executable='fourbar_joint_publisher.py',
        name='fourbar_joint_publisher',
        output='screen'
    )

    rviz_args = []
    if os.path.exists(rviz_config):
        rviz_args = ['-d', rviz_config]

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_args
    )

    return LaunchDescription([
        robot_state_publisher,
        fourbar_joint_publisher,
        rviz2,
    ])
