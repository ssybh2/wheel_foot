#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class LegEffortSymmetricHold(Node):
    def __init__(self):
        super().__init__('leg_effort_symmetric_hold')

        self.joint_names = [
            'big_arm_1_joint',
            'big_arm_2_joint',
            'big_arm_3_joint',
            'big_arm_4_joint',
        ]

        # Target joint angles, rad
        self.declare_parameter('j1', 0.0)
        self.declare_parameter('j2', 0.0)
        self.declare_parameter('j3', 0.0)
        self.declare_parameter('j4', 0.0)

        # Leg joint PD parameters
        self.declare_parameter('kp', 2.0)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('max_effort', 2.0)

        # Wait for robot to drop first
        self.declare_parameter('start_delay', 3.0)

        self.pos = {}
        self.vel = {}

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            20
        )

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/leg_effort_controller/commands',
            10
        )

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.timer = self.create_timer(0.01, self.loop)

        self.get_logger().info('leg_effort_symmetric_hold started.')
        self.get_logger().info('Publishing effort commands to /leg_effort_controller/commands')

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                if i < len(msg.position):
                    self.pos[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.vel[name] = msg.velocity[i]

    def ready(self):
        return all(name in self.pos for name in self.joint_names)

    def clamp(self, value, limit):
        if value > limit:
            return limit
        if value < -limit:
            return -limit
        return value

    def loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        start_delay = float(self.get_parameter('start_delay').value)

        if now - self.start_time < start_delay:
            return

        if not self.ready():
            return

        kp = float(self.get_parameter('kp').value)
        kd = float(self.get_parameter('kd').value)
        max_effort = float(self.get_parameter('max_effort').value)

        targets = [
            float(self.get_parameter('j1').value),
            float(self.get_parameter('j2').value),
            float(self.get_parameter('j3').value),
            float(self.get_parameter('j4').value),
        ]

        efforts = []

        for name, target in zip(self.joint_names, targets):
            p = self.pos.get(name, 0.0)
            v = self.vel.get(name, 0.0)

            error = target - p
            effort = kp * error - kd * v
            effort = self.clamp(effort, max_effort)

            efforts.append(effort)

        msg = Float64MultiArray()
        msg.data = efforts
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LegEffortSymmetricHold()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    try:
        stop = Float64MultiArray()
        stop.data = [0.0, 0.0, 0.0, 0.0]
        if rclpy.ok():
            node.pub.publish(stop)
    except Exception:
        pass

    node.destroy_node()

    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
