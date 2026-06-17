#!/usr/bin/env python3
"""Unified six-motor manual controller for the wheel-leg robot.

This node is the ONLY publisher to both controller command topics while it runs:

  /leg_effort_controller/commands
      [big_arm_1, big_arm_2, big_arm_3, big_arm_4] effort in N*m

  /wheel_effort_controller/commands
      [wheel_1, wheel_2] effort in N*m

Control architecture:
- Four leg joints: closed-loop PD hold around calibrated target angles.
- Two wheel joints: open-loop differential effort from keyboard commands.
- Releasing a motion key stops the wheels, but the leg PD remains active.

Keys:
  Hold W / S : forward / backward
  Hold A / D : turn left / right
  Space      : stop wheels immediately; leg hold remains active
  R / F      : raise / lower the symmetric leg extension target
  0          : reset leg extension offset
  ] / [      : increase / decrease drive torque
  = / -      : increase / decrease turn torque
  P          : print status
  H          : print help
  Q          : publish zero to all six joints and quit

Important:
- Stop leg_effort_symmetric_hold.py, lqr_balance_controller.py, and any other
  publisher to the two controller command topics before starting this node.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


HELP = """
Unified six-motor manual controller
===================================

Vehicle motion:
  Hold W : forward
  Hold S : backward
  Hold A : turn left
  Hold D : turn right
  SPACE  : stop wheels immediately; keep leg PD active

Leg posture:
  R / F  : raise / lower symmetric leg extension target
  0      : reset leg extension offset

Wheel torque:
  ] / [  : drive torque + / -
  = / -  : turn torque  + / -

Other:
  P      : print status
  H      : print help
  Q      : zero all six efforts and quit

Command order:
  legs   = [big_arm_1, big_arm_2, big_arm_3, big_arm_4]
  wheels = [wheel_1, wheel_2]
"""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SixMotorManualController(Node):
    LEG_JOINTS = [
        'big_arm_1_joint',
        'big_arm_2_joint',
        'big_arm_3_joint',
        'big_arm_4_joint',
    ]

    # Increasing this pattern follows the calibrated symmetric extension:
    # [a, -a, a, -a].
    EXTENSION_PATTERN = [1.0, -1.0, 1.0, -1.0]

    def __init__(self) -> None:
        super().__init__('six_motor_manual_controller')

        # Topics.
        self.declare_parameter(
            'wheel_command_topic',
            '/wheel_effort_controller/commands',
        )
        self.declare_parameter(
            'leg_command_topic',
            '/leg_effort_controller/commands',
        )
        self.declare_parameter('joint_state_topic', '/joint_states')

        # Main loop and keyboard deadman.
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('deadman_timeout', 0.70)
        self.declare_parameter('joint_state_timeout', 0.30)

        # Wheel effort settings.
        self.declare_parameter('drive_torque', 0.10)
        self.declare_parameter('turn_torque', 0.08)
        self.declare_parameter('wheel_torque_step', 0.02)
        self.declare_parameter('max_wheel_torque', 0.50)
        self.declare_parameter('wheel1_sign', 1.0)
        self.declare_parameter('wheel2_sign', 1.0)
        self.declare_parameter('forward_sign', 1.0)
        self.declare_parameter('turn_sign', 1.0)

        # Calibrated leg targets.
        self.declare_parameter('j1', 0.15)
        self.declare_parameter('j2', -0.15)
        self.declare_parameter('j3', 0.15)
        self.declare_parameter('j4', -0.15)

        # Leg PD.
        self.declare_parameter('leg_kp', 4.0)
        self.declare_parameter('leg_kd', 0.5)
        self.declare_parameter('max_leg_effort', 2.0)

        # Smooth target engagement and keyboard height adjustment.
        self.declare_parameter('start_delay', 0.50)
        self.declare_parameter('leg_target_slew_rate', 0.30)  # rad/s
        self.declare_parameter('leg_extension_step', 0.01)    # rad/key
        self.declare_parameter('max_leg_extension_offset', 0.20)

        self.wheel_topic = str(
            self.get_parameter('wheel_command_topic').value
        )
        self.leg_topic = str(
            self.get_parameter('leg_command_topic').value
        )
        joint_state_topic = str(
            self.get_parameter('joint_state_topic').value
        )

        self.publish_rate = max(
            10.0,
            float(self.get_parameter('publish_rate').value),
        )
        self.deadman_timeout = max(
            0.05,
            float(self.get_parameter('deadman_timeout').value),
        )
        self.joint_state_timeout = max(
            0.05,
            float(self.get_parameter('joint_state_timeout').value),
        )

        self.drive_torque = abs(
            float(self.get_parameter('drive_torque').value)
        )
        self.turn_torque = abs(
            float(self.get_parameter('turn_torque').value)
        )
        self.wheel_torque_step = max(
            0.001,
            abs(float(self.get_parameter('wheel_torque_step').value)),
        )
        self.max_wheel_torque = max(
            0.01,
            abs(float(self.get_parameter('max_wheel_torque').value)),
        )

        self.wheel1_sign = self.normalize_sign(
            float(self.get_parameter('wheel1_sign').value)
        )
        self.wheel2_sign = self.normalize_sign(
            float(self.get_parameter('wheel2_sign').value)
        )
        self.forward_sign = self.normalize_sign(
            float(self.get_parameter('forward_sign').value)
        )
        self.turn_sign = self.normalize_sign(
            float(self.get_parameter('turn_sign').value)
        )

        self.leg_extension_step = max(
            0.001,
            abs(float(self.get_parameter('leg_extension_step').value)),
        )
        self.max_leg_extension_offset = max(
            0.0,
            abs(
                float(
                    self.get_parameter(
                        'max_leg_extension_offset'
                    ).value
                )
            ),
        )

        self.position: Dict[str, float] = {}
        self.velocity: Dict[str, float] = {}
        self.last_joint_state_wall_time: Optional[float] = None

        self.commanded_leg_targets: Optional[List[float]] = None
        self.leg_extension_offset = 0.0

        self.motion_key: Optional[str] = None
        self.last_motion_key_wall_time = 0.0
        self.quit_requested = False
        self.armed = False

        self.start_wall_time = time.monotonic()
        self.last_control_wall_time = self.start_wall_time

        self.wheel_pub = self.create_publisher(
            Float64MultiArray,
            self.wheel_topic,
            10,
        )
        self.leg_pub = self.create_publisher(
            Float64MultiArray,
            self.leg_topic,
            10,
        )
        self.create_subscription(
            JointState,
            joint_state_topic,
            self.joint_state_callback,
            50,
        )

        print(HELP)
        self.print_status()
        self.publish_all_zero(repeat=3)

    @staticmethod
    def normalize_sign(value: float) -> float:
        return 1.0 if value >= 0.0 else -1.0

    def base_leg_targets(self) -> List[float]:
        return [
            float(self.get_parameter('j1').value),
            float(self.get_parameter('j2').value),
            float(self.get_parameter('j3').value),
            float(self.get_parameter('j4').value),
        ]

    def desired_leg_targets(self) -> List[float]:
        return [
            base + pattern * self.leg_extension_offset
            for base, pattern in zip(
                self.base_leg_targets(),
                self.EXTENSION_PATTERN,
            )
        ]

    def joint_state_callback(self, msg: JointState) -> None:
        received_any = False

        for index, name in enumerate(msg.name):
            if name not in self.LEG_JOINTS:
                continue

            if index < len(msg.position):
                self.position[name] = float(msg.position[index])
                received_any = True

            if index < len(msg.velocity):
                self.velocity[name] = float(msg.velocity[index])

        if received_any and all(
            name in self.position for name in self.LEG_JOINTS
        ):
            self.last_joint_state_wall_time = time.monotonic()

            if self.commanded_leg_targets is None:
                self.commanded_leg_targets = [
                    self.position[name]
                    for name in self.LEG_JOINTS
                ]

    def joint_states_ready(self) -> bool:
        return all(name in self.position for name in self.LEG_JOINTS)

    def joint_states_fresh(self, now: float) -> bool:
        if self.last_joint_state_wall_time is None:
            return False
        return (
            now - self.last_joint_state_wall_time
            <= self.joint_state_timeout
        )

    def wheel_command(self, now: float) -> Tuple[float, float]:
        if (
            self.motion_key is None
            or now - self.last_motion_key_wall_time
            > self.deadman_timeout
        ):
            self.motion_key = None
            return 0.0, 0.0

        if self.motion_key == 'w':
            value = self.forward_sign * self.drive_torque
            logical = (value, value)
        elif self.motion_key == 's':
            value = -self.forward_sign * self.drive_torque
            logical = (value, value)
        elif self.motion_key == 'a':
            value = self.turn_sign * self.turn_torque
            logical = (-value, value)
        elif self.motion_key == 'd':
            value = self.turn_sign * self.turn_torque
            logical = (value, -value)
        else:
            logical = (0.0, 0.0)

        wheel1 = clamp(
            logical[0] * self.wheel1_sign,
            -self.max_wheel_torque,
            self.max_wheel_torque,
        )
        wheel2 = clamp(
            logical[1] * self.wheel2_sign,
            -self.max_wheel_torque,
            self.max_wheel_torque,
        )
        return wheel1, wheel2

    def update_commanded_leg_targets(self, dt: float) -> None:
        if self.commanded_leg_targets is None:
            return

        desired = self.desired_leg_targets()
        slew_rate = max(
            0.01,
            abs(
                float(
                    self.get_parameter(
                        'leg_target_slew_rate'
                    ).value
                )
            ),
        )
        max_step = slew_rate * max(0.0, dt)

        for index, target in enumerate(desired):
            error = target - self.commanded_leg_targets[index]
            self.commanded_leg_targets[index] += clamp(
                error,
                -max_step,
                max_step,
            )

    def leg_efforts(self, now: float, dt: float) -> List[float]:
        if not self.joint_states_ready():
            return [0.0] * 4

        if not self.joint_states_fresh(now):
            return [0.0] * 4

        start_delay = max(
            0.0,
            float(self.get_parameter('start_delay').value),
        )
        if now - self.start_wall_time < start_delay:
            return [0.0] * 4

        self.update_commanded_leg_targets(dt)
        if self.commanded_leg_targets is None:
            return [0.0] * 4

        kp = float(self.get_parameter('leg_kp').value)
        kd = float(self.get_parameter('leg_kd').value)
        max_effort = max(
            0.0,
            abs(float(self.get_parameter('max_leg_effort').value)),
        )

        efforts: List[float] = []
        for name, target in zip(
            self.LEG_JOINTS,
            self.commanded_leg_targets,
        ):
            position = self.position[name]
            velocity = self.velocity.get(name, 0.0)
            effort = (
                kp * (target - position)
                - kd * velocity
            )
            efforts.append(
                clamp(effort, -max_effort, max_effort)
            )

        return efforts

    def publish_wheels(self, values: Tuple[float, float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(values[0]), float(values[1])]
        self.wheel_pub.publish(msg)

    def publish_legs(self, values: List[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(value) for value in values]
        self.leg_pub.publish(msg)

    def publish_all_zero(self, repeat: int = 1) -> None:
        for _ in range(max(1, repeat)):
            self.publish_wheels((0.0, 0.0))
            self.publish_legs([0.0, 0.0, 0.0, 0.0])
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.02)

    def control_once(self) -> None:
        now = time.monotonic()
        dt = clamp(
            now - self.last_control_wall_time,
            0.0,
            0.1,
        )
        self.last_control_wall_time = now

        if not self.joint_states_fresh(now):
            self.publish_wheels((0.0, 0.0))
            self.publish_legs([0.0, 0.0, 0.0, 0.0])
            return

        wheel_values = self.wheel_command(now)
        leg_values = self.leg_efforts(now, dt)

        self.publish_wheels(wheel_values)
        self.publish_legs(leg_values)

    def stop_wheels(self) -> None:
        self.motion_key = None
        self.publish_wheels((0.0, 0.0))

    def adjust_leg_extension(self, direction: float) -> None:
        self.leg_extension_offset = clamp(
            self.leg_extension_offset
            + direction * self.leg_extension_step,
            -self.max_leg_extension_offset,
            self.max_leg_extension_offset,
        )
        self.print_status()

    def handle_key(self, key: str) -> None:
        key = key.lower()

        if key in ('w', 'a', 's', 'd'):
            self.motion_key = key
            self.last_motion_key_wall_time = time.monotonic()
            return

        if key == ' ':
            self.stop_wheels()
            print('\nWheel stop; leg PD remains active.')
            return

        if key == 'r':
            self.adjust_leg_extension(+1.0)
            return

        if key == 'f':
            self.adjust_leg_extension(-1.0)
            return

        if key == '0':
            self.leg_extension_offset = 0.0
            self.print_status()
            return

        if key == ']':
            self.drive_torque = min(
                self.max_wheel_torque,
                self.drive_torque + self.wheel_torque_step,
            )
            self.print_status()
            return

        if key == '[':
            self.drive_torque = max(
                0.0,
                self.drive_torque - self.wheel_torque_step,
            )
            self.print_status()
            return

        if key == '=':
            self.turn_torque = min(
                self.max_wheel_torque,
                self.turn_torque + self.wheel_torque_step,
            )
            self.print_status()
            return

        if key == '-':
            self.turn_torque = max(
                0.0,
                self.turn_torque - self.wheel_torque_step,
            )
            self.print_status()
            return

        if key == 'p':
            self.print_status()
            return

        if key == 'h':
            print('\n' + HELP)
            self.print_status()
            return

        if key == 'q' or key == '\x03':
            self.quit_requested = True

    def print_status(self) -> None:
        targets = self.desired_leg_targets()
        print(
            '\nStatus: '
            f'drive={self.drive_torque:.3f} N*m, '
            f'turn={self.turn_torque:.3f} N*m, '
            f'leg_offset={self.leg_extension_offset:+.3f} rad, '
            'leg_targets=['
            + ', '.join(f'{value:+.3f}' for value in targets)
            + '], '
            f'wheel_signs=[{self.wheel1_sign:+.0f}, '
            f'{self.wheel2_sign:+.0f}]'
        )

    def run(self) -> None:
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())
            period = 1.0 / self.publish_rate
            next_control = time.monotonic()

            while rclpy.ok() and not self.quit_requested:
                now = time.monotonic()
                timeout = max(
                    0.0,
                    min(period, next_control - now),
                )

                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    timeout,
                )
                if readable:
                    self.handle_key(sys.stdin.read(1))

                rclpy.spin_once(self, timeout_sec=0.0)

                now = time.monotonic()
                if now >= next_control:
                    self.control_once()
                    next_control = now + period

        finally:
            # Restore the terminal immediately. Do not publish here: cleanup is
            # centralized in main(), while the ROS context is still valid.
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                old_settings,
            )


def main(args=None) -> None:
    # Disable rclpy's automatic SIGINT shutdown. Python still raises
    # KeyboardInterrupt, allowing us to publish zero effort before shutdown.
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = SixMotorManualController()

    try:
        node.run()
    except KeyboardInterrupt:
        print('\nCtrl-C received; stopping all six motors safely.')
    finally:
        if rclpy.ok():
            try:
                node.publish_all_zero(repeat=5)
                print('All six effort commands stopped at zero.')
            except Exception as exc:
                print(f'Warning: failed to publish final zero effort: {exc}')

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
