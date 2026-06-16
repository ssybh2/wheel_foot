#!/usr/bin/env python3
"""Keyboard teleoperation for a two-wheel effort controller.

Controls:
  Hold W : forward
  Hold S : backward
  Hold A : turn left in place
  Hold D : turn right in place
  Space  : immediate stop
  ] / [  : increase / decrease drive torque
  = / -  : increase / decrease turn torque
  P      : print current settings
  H      : help
  Q      : stop and quit

The node continuously publishes to:
  /wheel_effort_controller/commands

Safety:
- Commands automatically return to zero when no motion key has been received
  for ``deadman_timeout`` seconds.
- Space sends zero immediately.
- Exiting sends zero several times.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


HELP = """
Wheel effort keyboard teleop
============================

Hold keys:
  W : forward
  S : backward
  A : turn left in place
  D : turn right in place

Safety:
  SPACE : immediate stop
  Q     : stop and quit

Torque adjustment:
  ] / [ : drive torque + / -
  = / - : turn torque  + / -

Other:
  P : print settings
  H : print help

The displayed command order is:
  [wheel_1_joint, wheel_2_joint]
"""


class WheelKeyboardTeleop(Node):
    def __init__(self) -> None:
        super().__init__('wheel_keyboard_teleop')

        self.declare_parameter(
            'command_topic',
            '/wheel_effort_controller/commands',
        )
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('drive_torque', 0.10)
        self.declare_parameter('turn_torque', 0.08)
        self.declare_parameter('torque_step', 0.02)
        self.declare_parameter('max_torque', 0.50)
        self.declare_parameter('deadman_timeout', 0.70)

        # Use these parameters to correct joint-axis directions without
        # changing the source code. Allowed practical values are +1 or -1.
        self.declare_parameter('wheel1_sign', 1.0)
        self.declare_parameter('wheel2_sign', 1.0)

        # Change these only for operator convention:
        # forward_sign=-1 swaps W/S; turn_sign=-1 swaps A/D.
        self.declare_parameter('forward_sign', 1.0)
        self.declare_parameter('turn_sign', 1.0)

        self.command_topic = str(
            self.get_parameter('command_topic').value
        )
        self.publish_rate = max(
            1.0,
            float(self.get_parameter('publish_rate').value),
        )
        self.drive_torque = abs(
            float(self.get_parameter('drive_torque').value)
        )
        self.turn_torque = abs(
            float(self.get_parameter('turn_torque').value)
        )
        self.torque_step = max(
            0.001,
            abs(float(self.get_parameter('torque_step').value)),
        )
        self.max_torque = max(
            0.01,
            abs(float(self.get_parameter('max_torque').value)),
        )
        self.deadman_timeout = max(
            0.05,
            float(self.get_parameter('deadman_timeout').value),
        )
        self.wheel1_sign = self._normalize_sign(
            float(self.get_parameter('wheel1_sign').value)
        )
        self.wheel2_sign = self._normalize_sign(
            float(self.get_parameter('wheel2_sign').value)
        )
        self.forward_sign = self._normalize_sign(
            float(self.get_parameter('forward_sign').value)
        )
        self.turn_sign = self._normalize_sign(
            float(self.get_parameter('turn_sign').value)
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.motion_key: Optional[str] = None
        self.last_motion_key_time = 0.0
        self.last_published: Optional[Tuple[float, float]] = None
        self.quit_requested = False

        print(HELP)
        self.print_settings()
        self.publish_zero(repeat=3)

    @staticmethod
    def _normalize_sign(value: float) -> float:
        return 1.0 if value >= 0.0 else -1.0

    def clamp(self, value: float) -> float:
        return max(-self.max_torque, min(self.max_torque, value))

    def logical_command(self) -> Tuple[float, float]:
        """Return logical wheel torques before joint-direction correction."""
        now = time.monotonic()

        if (
            self.motion_key is None
            or now - self.last_motion_key_time > self.deadman_timeout
        ):
            self.motion_key = None
            return 0.0, 0.0

        if self.motion_key == 'w':
            value = self.forward_sign * self.drive_torque
            return value, value
        if self.motion_key == 's':
            value = -self.forward_sign * self.drive_torque
            return value, value
        if self.motion_key == 'a':
            value = self.turn_sign * self.turn_torque
            return -value, value
        if self.motion_key == 'd':
            value = self.turn_sign * self.turn_torque
            return value, -value

        return 0.0, 0.0

    def physical_command(self) -> Tuple[float, float]:
        wheel1, wheel2 = self.logical_command()

        wheel1 = self.clamp(wheel1 * self.wheel1_sign)
        wheel2 = self.clamp(wheel2 * self.wheel2_sign)
        return wheel1, wheel2

    def publish_current(self, force_print: bool = False) -> None:
        wheel1, wheel2 = self.physical_command()

        msg = Float64MultiArray()
        msg.data = [wheel1, wheel2]
        self.publisher.publish(msg)

        current = (wheel1, wheel2)
        if force_print or current != self.last_published:
            mode = self.motion_key.upper() if self.motion_key else 'STOP'
            print(
                f'\rmode={mode:>4s} | '
                f'cmd=[{wheel1:+.3f}, {wheel2:+.3f}] N*m | '
                f'drive={self.drive_torque:.3f} | '
                f'turn={self.turn_torque:.3f}      ',
                end='',
                flush=True,
            )
            self.last_published = current

    def publish_zero(self, repeat: int = 1) -> None:
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0]

        for _ in range(max(1, repeat)):
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.02)

        self.motion_key = None
        self.last_published = (0.0, 0.0)

    def handle_key(self, key: str) -> None:
        key = key.lower()

        if key in ('w', 'a', 's', 'd'):
            self.motion_key = key
            self.last_motion_key_time = time.monotonic()
            return

        if key == ' ':
            self.publish_zero(repeat=2)
            print('\nEmergency stop.')
            return

        if key == ']':
            self.drive_torque = min(
                self.max_torque,
                self.drive_torque + self.torque_step,
            )
            self.print_settings()
            return

        if key == '[':
            self.drive_torque = max(
                0.0,
                self.drive_torque - self.torque_step,
            )
            self.print_settings()
            return

        if key == '=':
            self.turn_torque = min(
                self.max_torque,
                self.turn_torque + self.torque_step,
            )
            self.print_settings()
            return

        if key == '-':
            self.turn_torque = max(
                0.0,
                self.turn_torque - self.torque_step,
            )
            self.print_settings()
            return

        if key == 'p':
            self.print_settings()
            return

        if key == 'h':
            print('\n' + HELP)
            self.print_settings()
            return

        if key == 'q' or key == '\x03':
            self.quit_requested = True
            return

    def print_settings(self) -> None:
        print(
            '\nSettings: '
            f'topic={self.command_topic}, '
            f'drive={self.drive_torque:.3f} N*m, '
            f'turn={self.turn_torque:.3f} N*m, '
            f'max={self.max_torque:.3f} N*m, '
            f'deadman={self.deadman_timeout:.2f} s, '
            f'wheel_signs=[{self.wheel1_sign:+.0f}, {self.wheel2_sign:+.0f}], '
            f'forward_sign={self.forward_sign:+.0f}, '
            f'turn_sign={self.turn_sign:+.0f}'
        )

    def run(self) -> None:
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())
            period = 1.0 / self.publish_rate
            next_publish = time.monotonic()

            while rclpy.ok() and not self.quit_requested:
                now = time.monotonic()
                wait_time = max(0.0, min(period, next_publish - now))

                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    wait_time,
                )
                if readable:
                    key = sys.stdin.read(1)
                    self.handle_key(key)

                now = time.monotonic()
                if now >= next_publish:
                    self.publish_current()
                    rclpy.spin_once(self, timeout_sec=0.0)
                    next_publish = now + period

        finally:
            self.publish_zero(repeat=5)
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                old_settings,
            )
            print('\nWheel command stopped at [0.0, 0.0].')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelKeyboardTeleop()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero(repeat=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
