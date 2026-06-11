#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


def quat_to_pitch(qx, qy, qz, qw):
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class BalanceStateMonitor(Node):
    def __init__(self):
        super().__init__('balance_state_monitor')

        self.pitch = None
        self.pitch_rate = None
        self.x = None
        self.x_dot = None
        self.wheel_1_vel = None
        self.wheel_2_vel = None

        self.create_subscription(Odometry, '/base_odom', self.odom_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        self.timer = self.create_timer(0.1, self.print_state)

        self.get_logger().info('balance_state_monitor started.')
        self.get_logger().info('Listening to /base_odom and /joint_states')

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation

        self.pitch = quat_to_pitch(q.x, q.y, q.z, q.w)
        self.pitch_rate = msg.twist.twist.angular.y

        self.x = msg.pose.pose.position.x
        self.x_dot = msg.twist.twist.linear.x

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name == 'wheel_1_joint':
                if i < len(msg.velocity):
                    self.wheel_1_vel = msg.velocity[i]

            elif name == 'wheel_2_joint':
                if i < len(msg.velocity):
                    self.wheel_2_vel = msg.velocity[i]

    def print_state(self):
        if self.pitch is None or self.wheel_1_vel is None or self.wheel_2_vel is None:
            return

        pitch_deg = math.degrees(self.pitch)
        pitch_rate_deg = math.degrees(self.pitch_rate)

        print(
            f"pitch={pitch_deg:+7.3f} deg | "
            f"pitch_rate={pitch_rate_deg:+8.3f} deg/s | "
            f"x={self.x:+7.3f} m | "
            f"x_dot={self.x_dot:+7.3f} m/s | "
            f"w1={self.wheel_1_vel:+7.3f} rad/s | "
            f"w2={self.wheel_2_vel:+7.3f} rad/s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BalanceStateMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
