#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


def quat_to_pitch(qx, qy, qz, qw):
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class SimpleBalanceController(Node):
    def __init__(self):
        super().__init__('simple_balance_controller')

        self.declare_parameter('kp', 2.0)
        self.declare_parameter('kd', 0.4)
        self.declare_parameter('max_speed', 4.0)
        self.declare_parameter('control_sign', 1.0)
        self.declare_parameter('fall_angle_deg', 35.0)

        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.has_odom = False

        self.sub_odom = self.create_subscription(
            Odometry,
            '/base_odom',
            self.odom_cb,
            20
        )

        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            '/wheel_velocity_controller/commands',
            10
        )

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('simple_balance_controller started.')
        self.get_logger().info('This is a safe PD test before real LQR.')

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.pitch = quat_to_pitch(q.x, q.y, q.z, q.w)
        self.pitch_rate = msg.twist.twist.angular.y
        self.has_odom = True

    def control_loop(self):
        if not self.has_odom:
            return

        kp = float(self.get_parameter('kp').value)
        kd = float(self.get_parameter('kd').value)
        max_speed = float(self.get_parameter('max_speed').value)
        control_sign = float(self.get_parameter('control_sign').value)
        fall_angle = math.radians(float(self.get_parameter('fall_angle_deg').value))

        cmd = Float64MultiArray()

        # Safety: if robot falls too much, stop wheels.
        if abs(self.pitch) > fall_angle:
            cmd.data = [0.0, 0.0]
            self.pub_cmd.publish(cmd)
            return

        # Simple PD balance test.
        u = control_sign * (kp * self.pitch + kd * self.pitch_rate)

        # Saturation.
        u = max(-max_speed, min(max_speed, u))

        cmd.data = [u, u]
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleBalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Stop wheels when exiting.
    stop_msg = Float64MultiArray()
    stop_msg.data = [0.0, 0.0]
    node.pub_cmd.publish(stop_msg)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
