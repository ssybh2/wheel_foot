#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def rot_y(v, q):
    c = math.cos(q)
    s = math.sin(q)
    x, y, z = v
    return (c * x + s * z, y, -s * x + c * z)


def angle_y(v):
    return math.atan2(-v[2], v[0])


def norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def dist_xz(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def circle_intersections_xz(c1, r1, c2, r2):
    x1, z1 = c1[0], c1[2]
    x2, z2 = c2[0], c2[2]

    dx = x2 - x1
    dz = z2 - z1
    d = math.hypot(dx, dz)

    if d < 1e-9:
        return []

    if d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    h = math.sqrt(max(0.0, h2))

    ex = dx / d
    ez = dz / d

    px = x1 + a * ex
    pz = z1 + a * ez

    ix1 = px - h * ez
    iz1 = pz + h * ex

    ix2 = px + h * ez
    iz2 = pz - h * ex

    return [(ix1, iz1), (ix2, iz2)]


class FourBarJointPublisher(Node):
    def __init__(self):
        super().__init__('fourbar_joint_publisher')

        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        self.sub_cmd = self.create_subscription(
            Float64MultiArray,
            '/fourbar_cmd',
            self.cmd_callback,
            10
        )

        # data = [q1, q2, q3, q4, wheel_1_speed, wheel_2_speed]
        self.cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.wheel_1_pos = 0.0
        self.wheel_2_pos = 0.0

        self.J1 = (0.231306,  0.053564, 0.030643)
        self.J2 = (0.161440,  0.053564, 0.026313)
        self.J3 = (0.231306, -0.048436, 0.030643)
        self.J4 = (0.161440, -0.048436, 0.026313)

        self.K1 = (0.308448,  0.080564,  0.009451)
        self.K2 = (0.091299,  0.088564, -0.012160)
        self.K3 = (0.306005, -0.083436,  0.002006)
        self.K4 = (0.090381, -0.075436, -0.010437)

        self.W1 = (0.204820,  0.080564, -0.051057)
        self.W2 = (0.201207, -0.083436, -0.056453)

        # Group 1:
        # big_arm_2 -> small_leg_wheel_1 -> wheel_motor_1
        # big_arm_3 -> small_leggy_1
        self.group1 = {
            'JA': self.J2,
            'KA0': self.K2,
            'JB': self.J3,
            'KB0': self.K1,
            'W0': self.W1,
        }

        # Group 2:
        # big_arm_1 -> small_leg_wheel_2 -> wheel_motor_2
        # big_arm_4 -> small_leggy_2
        self.group2 = {
            'JA': self.J1,
            'KA0': self.K3,
            'JB': self.J4,
            'KB0': self.K4,
            'W0': self.W2,
        }

        self.prev_w1_xz = (self.W1[0], self.W1[2])
        self.prev_w2_xz = (self.W2[0], self.W2[2])

        self.last_time = self.get_clock().now()

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.get_logger().info('fourbar_joint_publisher command mode started.')
        self.get_logger().info('Subscribe topic: /fourbar_cmd')
        self.get_logger().info('data = [q1, q2, q3, q4, wheel_1_speed, wheel_2_speed]')

    def cmd_callback(self, msg):
        if len(msg.data) < 6:
            self.get_logger().warn(
                'Need 6 numbers: [q1, q2, q3, q4, wheel_1_speed, wheel_2_speed]'
            )
            return

        self.cmd = [float(v) for v in msg.data[:6]]

    def solve_group(self, group, qA, qB, prev_w_xz):
        JA = group['JA']
        KA0 = group['KA0']
        JB = group['JB']
        KB0 = group['KB0']
        W0 = group['W0']

        KA = add(JA, rot_y(sub(KA0, JA), qA))
        KB = add(JB, rot_y(sub(KB0, JB), qB))

        rA = dist_xz(W0, KA0)
        rB = dist_xz(W0, KB0)

        sols = circle_intersections_xz(KA, rA, KB, rB)

        if not sols:
            wx, wz = prev_w_xz
        else:
            def cost(p):
                return (p[0] - prev_w_xz[0]) ** 2 + (p[1] - prev_w_xz[1]) ** 2
            wx, wz = min(sols, key=cost)

        W = (wx, W0[1], wz)

        vA0 = sub(W0, KA0)
        vA = sub(W, KA)
        total_A = norm_angle(angle_y(vA) - angle_y(vA0))
        knee_A = norm_angle(total_A - qA)

        vB0 = sub(W0, KB0)
        vB = sub(W, KB)
        total_B = norm_angle(angle_y(vB) - angle_y(vB0))
        knee_B = norm_angle(total_B - qB)

        return knee_A, knee_B, (wx, wz)

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        q1, q2, q3, q4, wheel_1_speed, wheel_2_speed = self.cmd

        self.wheel_1_pos += wheel_1_speed * dt
        self.wheel_2_pos += wheel_2_speed * dt

        knee2, knee3, self.prev_w1_xz = self.solve_group(
            self.group1,
            q2,
            q3,
            self.prev_w1_xz
        )

        knee1, knee4, self.prev_w2_xz = self.solve_group(
            self.group2,
            q1,
            q4,
            self.prev_w2_xz
        )

        msg = JointState()
        msg.header.stamp = now.to_msg()

        msg.name = [
            'big_arm_1_joint',
            'big_arm_2_joint',
            'big_arm_3_joint',
            'big_arm_4_joint',
            'knee_1_joint',
            'knee_2_joint',
            'knee_3_joint',
            'knee_4_joint',
            'wheel_1_joint',
            'wheel_2_joint',
        ]

        msg.position = [
            q1,
            q2,
            q3,
            q4,
            knee1,
            knee2,
            knee3,
            knee4,
            self.wheel_1_pos,
            self.wheel_2_pos,
        ]

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FourBarJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
