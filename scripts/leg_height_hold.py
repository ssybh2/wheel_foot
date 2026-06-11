#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class LegHeightHold(Node):
    def __init__(self):
        super().__init__('leg_height_hold')

        self.joint_names = [
            'big_arm_1_joint',
            'big_arm_2_joint',
            'big_arm_3_joint',
            'big_arm_4_joint',
        ]

        self.declare_parameter('use_current_as_target', True)
        self.declare_parameter('j1', 0.0)
        self.declare_parameter('j2', 0.0)
        self.declare_parameter('j3', 0.0)
        self.declare_parameter('j4', 0.0)

        self.current_pos = {}
        self.target = None

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            20
        )

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/leg_position_controller/commands',
            10
        )

        self.timer = self.create_timer(0.02, self.loop)

        self.get_logger().info('leg_height_hold started.')
        self.get_logger().info('Holding big_arm_1/2/3/4 positions.')

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names and i < len(msg.position):
                self.current_pos[name] = msg.position[i]

    def init_target(self):
        use_current = bool(self.get_parameter('use_current_as_target').value)

        if use_current:
            if not all(name in self.current_pos for name in self.joint_names):
                return False

            self.target = [
                self.current_pos['big_arm_1_joint'],
                self.current_pos['big_arm_2_joint'],
                self.current_pos['big_arm_3_joint'],
                self.current_pos['big_arm_4_joint'],
            ]

            self.get_logger().info(
                'Using current big arm positions as target: '
                + ', '.join([f'{v:+.4f}' for v in self.target])
            )

        else:
            self.target = [
                float(self.get_parameter('j1').value),
                float(self.get_parameter('j2').value),
                float(self.get_parameter('j3').value),
                float(self.get_parameter('j4').value),
            ]

            self.get_logger().info(
                'Using parameter targets: '
                + ', '.join([f'{v:+.4f}' for v in self.target])
            )

        return True

    def loop(self):
        if self.target is None:
            if not self.init_target():
                return

        msg = Float64MultiArray()
        msg.data = self.target
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LegHeightHold()

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
