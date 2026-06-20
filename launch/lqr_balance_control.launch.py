from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('j1', default_value='0.08'),
        DeclareLaunchArgument('j2', default_value='-0.08'),
        DeclareLaunchArgument('j3', default_value='0.08'),
        DeclareLaunchArgument('j4', default_value='-0.08'),
        DeclareLaunchArgument('leg_kp', default_value='2.0'),
        DeclareLaunchArgument('leg_kd', default_value='0.3'),
        DeclareLaunchArgument('leg_max_effort', default_value='10.0'),
        DeclareLaunchArgument('leg_start_delay', default_value='0.0'),
        DeclareLaunchArgument('M', default_value='1.5'),
        DeclareLaunchArgument('m', default_value='3.0'),
        DeclareLaunchArgument('l', default_value='0.16'),
        DeclareLaunchArgument('q_theta', default_value='80.0'),
        DeclareLaunchArgument('q_theta_dot', default_value='8.0'),
        DeclareLaunchArgument('q_x', default_value='0.5'),
        DeclareLaunchArgument('q_x_dot', default_value='2.0'),
        DeclareLaunchArgument('r_u', default_value='1.0'),
        DeclareLaunchArgument('control_sign', default_value='1.0'),
        DeclareLaunchArgument('max_wheel_torque', default_value='2.0'),
        DeclareLaunchArgument('fall_angle_deg', default_value='60.0'),
        DeclareLaunchArgument('pitch_zero_mode', default_value='world'),
        DeclareLaunchArgument('pitch_zero_deg', default_value='0.0'),
        DeclareLaunchArgument('pitch_sign', default_value='1.0'),
    ]

    leg_hold = Node(
        package='wheel_leg_description',
        executable='leg_effort_symmetric_hold.py',
        name='leg_effort_symmetric_hold',
        output='screen',
        parameters=[{
            'j1': LaunchConfiguration('j1'),
            'j2': LaunchConfiguration('j2'),
            'j3': LaunchConfiguration('j3'),
            'j4': LaunchConfiguration('j4'),
            'kp': LaunchConfiguration('leg_kp'),
            'kd': LaunchConfiguration('leg_kd'),
            'max_effort': LaunchConfiguration('leg_max_effort'),
            'start_delay': LaunchConfiguration('leg_start_delay'),
        }],
    )

    lqr_balance = Node(
        package='wheel_leg_description',
        executable='lqr_balance_controller.py',
        name='lqr_balance_controller',
        output='screen',
        parameters=[{
            'M': LaunchConfiguration('M'),
            'm': LaunchConfiguration('m'),
            'l': LaunchConfiguration('l'),
            'q_theta': LaunchConfiguration('q_theta'),
            'q_theta_dot': LaunchConfiguration('q_theta_dot'),
            'q_x': LaunchConfiguration('q_x'),
            'q_x_dot': LaunchConfiguration('q_x_dot'),
            'r_u': LaunchConfiguration('r_u'),
            'control_sign': LaunchConfiguration('control_sign'),
            'max_wheel_torque': LaunchConfiguration('max_wheel_torque'),
            'fall_angle_deg': LaunchConfiguration('fall_angle_deg'),
            'pitch_zero_mode': LaunchConfiguration('pitch_zero_mode'),
            'pitch_zero_deg': LaunchConfiguration('pitch_zero_deg'),
            'pitch_sign': LaunchConfiguration('pitch_sign'),
        }],
    )

    return LaunchDescription(args + [leg_hold, lqr_balance])
