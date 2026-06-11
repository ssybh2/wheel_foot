#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


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
    # Rotation angle around Y axis, using X-Z plane.
    # This convention matches URDF <axis xyz="0 1 0"/>.
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
    # Use only X-Z plane.
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

    # perpendicular in X-Z plane
    ix1 = px - h * ez
    iz1 = pz + h * ex

    ix2 = px + h * ez
    iz2 = pz - h * ex

    return [(ix1, iz1), (ix2, iz2)]


class FourBarJointPublisher(Node):
    def __init__(self):
        super().__init__('fourbar_joint_publisher')

        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        self.declare_parameter('amplitude', 0.35)
        self.declare_parameter('speed', 0.8)
        self.declare_parameter('wheel_speed', 4.0)

        # ==========================================================
        # Global URDF coordinates, unit: m
        # ==========================================================

        # Big arm root joints
        self.J1 = (0.231306,  0.053564, 0.030643)
        self.J2 = (0.161440,  0.053564, 0.026313)
        self.J3 = (0.231306, -0.048436, 0.030643)
        self.J4 = (0.161440, -0.048436, 0.026313)

        # Knee / leg joints
        self.K1 = (0.308448,  0.080564,  0.009451)
        self.K2 = (0.091299,  0.088564, -0.012160)
        self.K3 = (0.306005, -0.083436,  0.002006)
        self.K4 = (0.090381, -0.075436, -0.010437)

        # Wheel axes
        self.W1 = (0.204820,  0.080564, -0.051057)
        self.W2 = (0.201207, -0.083436, -0.056453)

        # Current corrected grouping:
        #
        # Group 1:
        #   green big_arm_2 -> orange small_leg_wheel_1 -> wheel_motor_1
        #   red   big_arm_3 -> magenta small_leggy_1     -> same W1
        #
        # Group 2:
        #   blue   big_arm_1 -> cyan small_leg_wheel_2 -> wheel_motor_2
        #   yellow big_arm_4 -> yellow small_leggy_2   -> same W2

        self.group1 = {
            'JA': self.J2,
            'KA0': self.K2,
            'JB': self.J3,
            'KB0': self.K1,
            'W0': self.W1,
        }

        self.group2 = {
            'JA': self.J1,
            'KA0': self.K3,
            'JB': self.J4,
            'KB0': self.K4,
            'W0': self.W2,
        }

        self.prev_w1_xz = (self.W1[0], self.W1[2])
        self.prev_w2_xz = (self.W2[0], self.W2[2])

        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.get_logger().info('fourbar_joint_publisher started.')
        self.get_logger().info('Do NOT run joint_state_publisher_gui at the same time.')

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
            # If geometry temporarily fails, keep previous W point.
            wx, wz = prev_w_xz
        else:
            # Choose the solution closest to previous W to avoid jumping.
            def cost(p):
                return (p[0] - prev_w_xz[0]) ** 2 + (p[1] - prev_w_xz[1]) ** 2

            wx, wz = min(sols, key=cost)

        W = (wx, W0[1], wz)

        # Solve knee angles.
        # Child link total orientation = big_arm_angle + knee_angle.
        # Therefore:
        # knee = desired_total_rotation - big_arm_angle

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
        t = (now - self.start_time).nanoseconds * 1e-9

        amp = float(self.get_parameter('amplitude').value)
        speed = float(self.get_parameter('speed').value)
        wheel_speed = float(self.get_parameter('wheel_speed').value)

        s = math.sin(speed * t)

        # Demo active big-arm motion.
        # 后面接真实控制时，这四个角度会来自电机/控制器，而不是 sin。
        q1 = amp * s
        q4 = -amp * s

        q2 = -amp * s
        q3 = amp * s

        # Group 1: big_arm_2 + big_arm_3 -> W1
        knee2, knee3, self.prev_w1_xz = self.solve_group(
            self.group1,
            q2,
            q3,
            self.prev_w1_xz
        )

        # Group 2: big_arm_1 + big_arm_4 -> W2
        knee1, knee4, self.prev_w2_xz = self.solve_group(
            self.group2,
            q1,
            q4,
            self.prev_w2_xz
        )

        wheel1 = wheel_speed * t
        wheel2 = wheel_speed * t

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
            wheel1,
            wheel2,
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
