#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

from scipy.linalg import solve_continuous_are


def quat_to_pitch(qx, qy, qz, qw):
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


def compute_lqr_gain(M, m, l, g, Q_diag, R_value):
    """
    Linear inverted pendulum on cart.

    State:
      x = [theta, theta_dot, position, velocity]

    Input:
      u = horizontal force, approximately

    theta = body pitch angle from upright.
    """

    A = np.array([
        [0.0,                 1.0, 0.0, 0.0],
        [(M + m) * g / (M*l), 0.0, 0.0, 0.0],
        [0.0,                 0.0, 0.0, 1.0],
        [-m * g / M,          0.0, 0.0, 0.0],
    ])

    B = np.array([
        [0.0],
        [-1.0 / (M*l)],
        [0.0],
        [1.0 / M],
    ])

    Q = np.diag(Q_diag)
    R = np.array([[R_value]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    return K


class LQRBalanceController(Node):
    def __init__(self):
        super().__init__('lqr_balance_controller')

        # Approximate inverted pendulum parameters.
        # You will tune these later using CAD / Gazebo values.
        self.declare_parameter('M', 1.5)       # lower body / wheel equivalent mass, kg
        self.declare_parameter('m', 3.0)       # upper body mass, kg
        self.declare_parameter('l', 0.16)      # COM height above wheel axle, m
        self.declare_parameter('g', 9.81)

        # LQR cost weights: [theta, theta_dot, x, x_dot]
        self.declare_parameter('q_theta', 80.0)
        self.declare_parameter('q_theta_dot', 8.0)
        self.declare_parameter('q_x', 0.5)
        self.declare_parameter('q_x_dot', 2.0)
        self.declare_parameter('r_u', 1.0)

        # Because current Gazebo controller accepts wheel velocity, not force.
        self.declare_parameter('force_to_wheel_speed', 0.8)
        self.declare_parameter('control_sign', 1.0)
        self.declare_parameter('max_wheel_speed', 8.0)

        self.declare_parameter('fall_angle_deg', 35.0)

        self.pitch = None
        self.pitch_rate = None
        self.x = None
        self.x_dot = None

        self.pitch0 = None
        self.x0 = None

        self.K = self.build_lqr()

        self.create_subscription(Odometry, '/base_odom', self.odom_cb, 20)

        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            '/wheel_velocity_controller/commands',
            10
        )

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('lqr_balance_controller started.')
        self.get_logger().info(f'LQR K = {self.K}')

    def build_lqr(self):
        M = float(self.get_parameter('M').value)
        m = float(self.get_parameter('m').value)
        l = float(self.get_parameter('l').value)
        g = float(self.get_parameter('g').value)

        Q_diag = [
            float(self.get_parameter('q_theta').value),
            float(self.get_parameter('q_theta_dot').value),
            float(self.get_parameter('q_x').value),
            float(self.get_parameter('q_x_dot').value),
        ]

        R_value = float(self.get_parameter('r_u').value)

        return compute_lqr_gain(M, m, l, g, Q_diag, R_value)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation

        self.pitch = quat_to_pitch(q.x, q.y, q.z, q.w)
        self.pitch_rate = msg.twist.twist.angular.y
        self.x = msg.pose.pose.position.x
        self.x_dot = msg.twist.twist.linear.x

        if self.pitch0 is None:
            self.pitch0 = self.pitch
            self.x0 = self.x
            self.get_logger().info(
                f'Set equilibrium: pitch0={math.degrees(self.pitch0):.4f} deg, x0={self.x0:.4f} m'
            )

    def control_loop(self):
        if self.pitch is None or self.pitch0 is None:
            return

        fall_angle = math.radians(float(self.get_parameter('fall_angle_deg').value))
        max_wheel_speed = float(self.get_parameter('max_wheel_speed').value)
        force_to_wheel_speed = float(self.get_parameter('force_to_wheel_speed').value)
        control_sign = float(self.get_parameter('control_sign').value)

        theta = self.pitch - self.pitch0
        theta_dot = self.pitch_rate
        x = self.x - self.x0
        x_dot = self.x_dot

        cmd = Float64MultiArray()

        if abs(theta) > fall_angle:
            cmd.data = [0.0, 0.0]
            self.pub_cmd.publish(cmd)
            return

        state = np.array([[theta], [theta_dot], [x], [x_dot]])

        # Standard LQR law.
        force_like_u = float(-(self.K @ state)[0, 0])

        # Current controller expects wheel angular speed, so map force-like u to rad/s.
        wheel_speed = control_sign * force_to_wheel_speed * force_like_u

        wheel_speed = max(-max_wheel_speed, min(max_wheel_speed, wheel_speed))

        cmd.data = [wheel_speed, wheel_speed]
        self.pub_cmd.publish(cmd)

    def stop(self):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0]
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LQRBalanceController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop()
    node.destroy_node()

    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
