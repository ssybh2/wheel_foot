#!/usr/bin/env python3
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


HELP = """
Keyboard control for wheel-leg fourbar model

Angles:
  q / a : big_arm_1 + / -
  w / s : big_arm_2 + / -
  e / d : big_arm_3 + / -
  r / f : big_arm_4 + / -

Wheel speeds:
  z / x : wheel_1 speed - / +
  c / v : wheel_2 speed - / +

Other:
  space : stop wheels
  0     : reset all angles and wheel speeds
  h     : print help
  Ctrl-C: exit
"""


class FourBarKeyboardCmd(Node):
    def __init__(self):
        super().__init__('fourbar_keyboard_cmd')

        self.pub = self.create_publisher(Float64MultiArray, '/fourbar_cmd', 10)

        self.angle_step = 0.04
        self.speed_step = 0.5

        # [q1, q2, q3, q4, wheel_1_speed, wheel_2_speed]
        self.cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.get_logger().info('fourbar_keyboard_cmd started.')
        print(HELP)
        self.publish_cmd()

    def publish_cmd(self):
        msg = Float64MultiArray()
        msg.data = self.cmd
        self.pub.publish(msg)

        print(
            f"\rq1={self.cmd[0]: .2f}, q2={self.cmd[1]: .2f}, "
            f"q3={self.cmd[2]: .2f}, q4={self.cmd[3]: .2f}, "
            f"w1={self.cmd[4]: .2f}, w2={self.cmd[5]: .2f}      ",
            end='',
            flush=True
        )

    def handle_key(self, key):
        if key == 'q':
            self.cmd[0] += self.angle_step
        elif key == 'a':
            self.cmd[0] -= self.angle_step

        elif key == 'w':
            self.cmd[1] += self.angle_step
        elif key == 's':
            self.cmd[1] -= self.angle_step

        elif key == 'e':
            self.cmd[2] += self.angle_step
        elif key == 'd':
            self.cmd[2] -= self.angle_step

        elif key == 'r':
            self.cmd[3] += self.angle_step
        elif key == 'f':
            self.cmd[3] -= self.angle_step

        elif key == 'z':
            self.cmd[4] -= self.speed_step
        elif key == 'x':
            self.cmd[4] += self.speed_step

        elif key == 'c':
            self.cmd[5] -= self.speed_step
        elif key == 'v':
            self.cmd[5] += self.speed_step

        elif key == ' ':
            self.cmd[4] = 0.0
            self.cmd[5] = 0.0

        elif key == '0':
            self.cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif key == 'h':
            print(HELP)

        # 安全限幅，防止四连杆解算跑飞
        for i in range(4):
            self.cmd[i] = max(-0.8, min(0.8, self.cmd[i]))

        self.publish_cmd()


def get_key():
    return sys.stdin.read(1)


def main(args=None):
    rclpy.init(args=args)
    node = FourBarKeyboardCmd()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            key = get_key()
            if key == '\x03':
                break
            node.handle_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
