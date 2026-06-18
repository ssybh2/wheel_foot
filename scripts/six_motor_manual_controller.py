#!/usr/bin/env python3
"""Unified six-motor keyboard teleoperation + balance controller.

This node is the sole command publisher for:

  /leg_effort_controller/commands
      [big_arm_1, big_arm_2, big_arm_3, big_arm_4] effort, N*m

  /wheel_effort_controller/commands
      [wheel_1, wheel_2] effort, N*m

Architecture
------------
1. Four leg motors:
   Joint-position PD keeps the calibrated symmetric leg configuration.

2. Motion generation:
   W/S request a target forward velocity.
   Velocity error generates a body pitch reference.

3. Drive modes:
   wheel_balance:
      Pitch/velocity feedback generates the common wheel effort.

   leg_lean_assist:
      The four big-arm motors move to a lean posture that should tilt the
      body and shift the center of mass. Wheel effort is reduced to an assist
      term so the wheels follow the moving body rather than being the primary
      drive actuator.

4. Steering:
   A/D add differential wheel effort.

The controller uses /base_odom for:
  pitch, pitch_rate, forward velocity, and forward position.

Safety
------
- Wheel effort is zero until both /joint_states and /base_odom are available.
- Stale odometry stops the wheels.
- Stale joint states stop all six efforts.
- Excessive pitch stops the wheels.
- SPACE stops motion while keeping leg PD active.
- Q or Ctrl-C sends zero to all six motors before ROS shutdown.
"""

from __future__ import annotations

import math
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


HELP = """
Six-motor balance teleoperation
===============================

Vehicle:
  Hold W : request forward velocity
  Hold S : request backward velocity
  Hold A : turn left
  Hold D : turn right
  SPACE  : target speed and turn command -> zero

Leg posture:
  R / F  : increase / decrease symmetric leg extension
  0      : reset leg extension offset

Speed and steering:
  ] / [  : increase / decrease commanded speed magnitude
  = / -  : increase / decrease steering effort

Other:
  P      : print controller status
  H      : print help
  Q      : zero all six motor efforts and quit

Command order:
  legs   = [big_arm_1, big_arm_2, big_arm_3, big_arm_4]
  wheels = [wheel_1, wheel_2]
"""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_sign(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


def sign_or_zero(value: float) -> float:
    if abs(value) < 1e-9:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


def quaternion_to_pitch(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> float:
    """Return pitch in radians using the standard ROS quaternion convention."""
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    if abs(sin_pitch) >= 1.0:
        return math.copysign(math.pi / 2.0, sin_pitch)
    return math.asin(sin_pitch)


class SixMotorBalanceTeleop(Node):
    LEG_JOINTS = [
        'big_arm_1_joint',
        'big_arm_2_joint',
        'big_arm_3_joint',
        'big_arm_4_joint',
    ]

    # Symmetric extension pattern:
    # [q1, q2, q3, q4] += offset * [1, -1, 1, -1]
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
        self.declare_parameter('odom_topic', '/base_odom')

        # Timing and safety.
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('deadman_timeout', 0.70)
        self.declare_parameter('joint_state_timeout', 0.30)
        self.declare_parameter('odom_timeout', 0.30)
        self.declare_parameter('start_delay', 0.50)
        self.declare_parameter('fall_angle_deg', 20.0)

        # Keyboard target-velocity command.
        self.declare_parameter('command_speed', 0.15)
        self.declare_parameter('speed_step', 0.05)
        self.declare_parameter('max_target_speed', 0.50)
        self.declare_parameter('target_speed_slew_rate', 0.60)

        # Speed -> pitch reference.
        self.declare_parameter('speed_to_pitch_gain', 0.80)
        self.declare_parameter('max_pitch_ref_deg', 6.0)
        self.declare_parameter('pitch_ref_sign', 1.0)

        # Drive architecture.
        self.declare_parameter('drive_mode', 'leg_lean_assist')

        # Leg-driven body lean. Positive pitch error adds
        # leg_lean_gain * (pitch_ref - pitch) * leg_lean_jN to each big-arm
        # target.
        self.declare_parameter('leg_lean_gain', 1.0)
        self.declare_parameter('max_leg_lean_offset', 0.25)
        self.declare_parameter('leg_lean_j1', 1.0)
        self.declare_parameter('leg_lean_j2', -1.0)
        self.declare_parameter('leg_lean_j3', 1.0)
        self.declare_parameter('leg_lean_j4', -1.0)
        self.declare_parameter('pitch_error_deadband_deg', 0.5)
        self.declare_parameter('leg_lean_effort_gain', 0.0)
        self.declare_parameter('max_leg_lean_effort', 0.0)

        # Wheel effort used in leg_lean_assist mode.
        self.declare_parameter('wheel_assist_pitch_kp', 0.5)
        self.declare_parameter('wheel_assist_pitch_kd', 0.10)
        self.declare_parameter('wheel_assist_velocity_kd', 0.20)
        self.declare_parameter('max_wheel_assist_torque', 0.60)

        # Balance / velocity feedback.
        self.declare_parameter('balance_kp', 8.0)
        self.declare_parameter('balance_kd', 0.8)
        self.declare_parameter('velocity_kd', 0.5)
        self.declare_parameter('position_hold_kp', 0.0)
        self.declare_parameter('balance_sign', 1.0)
        self.declare_parameter('pitch_sign', 1.0)

        # Steering and wheel limits.
        self.declare_parameter('turn_torque', 0.15)
        self.declare_parameter('turn_torque_step', 0.02)
        self.declare_parameter('max_turn_torque', 0.50)
        self.declare_parameter('max_wheel_torque', 1.00)
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
        self.declare_parameter('leg_kp', 2.0)
        self.declare_parameter('leg_kd', 0.20)
        self.declare_parameter('max_leg_effort', 3.0)
        self.declare_parameter('leg_target_slew_rate', 0.30)
        self.declare_parameter('leg_extension_step', 0.01)
        self.declare_parameter('max_leg_extension_offset', 0.20)

        # Status output.
        self.declare_parameter('status_period', 0.20)

        self.wheel_topic = str(
            self.get_parameter('wheel_command_topic').value
        )
        self.leg_topic = str(
            self.get_parameter('leg_command_topic').value
        )
        joint_state_topic = str(
            self.get_parameter('joint_state_topic').value
        )
        odom_topic = str(self.get_parameter('odom_topic').value)

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
        self.odom_timeout = max(
            0.05,
            float(self.get_parameter('odom_timeout').value),
        )

        self.command_speed = abs(
            float(self.get_parameter('command_speed').value)
        )
        self.speed_step = max(
            0.001,
            abs(float(self.get_parameter('speed_step').value)),
        )
        self.max_target_speed = max(
            0.0,
            abs(float(self.get_parameter('max_target_speed').value)),
        )

        self.turn_torque = abs(
            float(self.get_parameter('turn_torque').value)
        )
        self.turn_torque_step = max(
            0.001,
            abs(float(self.get_parameter('turn_torque_step').value)),
        )
        self.max_turn_torque = max(
            0.0,
            abs(float(self.get_parameter('max_turn_torque').value)),
        )
        self.max_wheel_torque = max(
            0.01,
            abs(float(self.get_parameter('max_wheel_torque').value)),
        )

        self.wheel1_sign = normalize_sign(
            float(self.get_parameter('wheel1_sign').value)
        )
        self.wheel2_sign = normalize_sign(
            float(self.get_parameter('wheel2_sign').value)
        )
        self.forward_sign = normalize_sign(
            float(self.get_parameter('forward_sign').value)
        )
        self.turn_sign = normalize_sign(
            float(self.get_parameter('turn_sign').value)
        )
        self.balance_sign = normalize_sign(
            float(self.get_parameter('balance_sign').value)
        )
        self.pitch_ref_sign = normalize_sign(
            float(self.get_parameter('pitch_ref_sign').value)
        )
        self.pitch_sign = normalize_sign(
            float(self.get_parameter('pitch_sign').value)
        )
        self.drive_mode = str(
            self.get_parameter('drive_mode').value
        ).strip().lower()
        if self.drive_mode not in ('wheel_balance', 'leg_lean_assist'):
            self.get_logger().warning(
                f'Unknown drive_mode={self.drive_mode!r}; '
                'falling back to leg_lean_assist.'
            )
            self.drive_mode = 'leg_lean_assist'

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

        # Joint feedback.
        self.leg_position: Dict[str, float] = {}
        self.leg_velocity: Dict[str, float] = {}
        self.last_joint_state_wall_time: Optional[float] = None

        # Odometry / body state.
        self.pitch: Optional[float] = None
        self.pitch_rate: Optional[float] = None
        self.position_x: Optional[float] = None
        self.velocity_x: Optional[float] = None
        self.last_odom_wall_time: Optional[float] = None

        # Automatically captured equilibrium.
        self.pitch0: Optional[float] = None
        self.position_hold_reference: Optional[float] = None

        # Leg target management.
        self.commanded_leg_targets: Optional[List[float]] = None
        self.leg_extension_offset = 0.0

        # Keyboard and motion command.
        self.motion_key: Optional[str] = None
        self.last_motion_key_wall_time = 0.0
        self.requested_target_speed = 0.0
        self.filtered_target_speed = 0.0
        self.requested_turn_effort = 0.0

        # Runtime.
        self.quit_requested = False
        self.start_wall_time = time.monotonic()
        self.last_control_wall_time = self.start_wall_time
        self.last_status_wall_time = 0.0
        self.last_pitch_ref = 0.0
        self.last_pitch_error = 0.0
        self.last_drive_effort = 0.0
        self.last_leg_lean_offset = 0.0
        self.last_leg_lean_effort = 0.0
        self.last_wheel_command = (0.0, 0.0)
        self.last_leg_command = [0.0, 0.0, 0.0, 0.0]
        self.last_safety_reason = 'waiting for feedback'

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
        self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            50,
        )

        print(HELP)
        self.print_configuration()
        self.publish_all_zero(repeat=3)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def joint_state_callback(self, msg: JointState) -> None:
        received = False

        for index, name in enumerate(msg.name):
            if name not in self.LEG_JOINTS:
                continue

            if index < len(msg.position):
                self.leg_position[name] = float(msg.position[index])
                received = True

            if index < len(msg.velocity):
                self.leg_velocity[name] = float(msg.velocity[index])

        if received and all(
            name in self.leg_position for name in self.LEG_JOINTS
        ):
            self.last_joint_state_wall_time = time.monotonic()

            if self.commanded_leg_targets is None:
                self.commanded_leg_targets = [
                    self.leg_position[name]
                    for name in self.LEG_JOINTS
                ]

    def odom_callback(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        measured_pitch = quaternion_to_pitch(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        self.pitch = self.pitch_sign * measured_pitch
        self.pitch_rate = (
            self.pitch_sign * float(msg.twist.twist.angular.y)
        )
        self.position_x = float(msg.pose.pose.position.x)
        self.velocity_x = float(msg.twist.twist.linear.x)
        self.last_odom_wall_time = time.monotonic()

        if self.pitch0 is None:
            self.pitch0 = self.pitch
            self.position_hold_reference = self.position_x
            self.get_logger().info(
                'Captured equilibrium: '
                f'pitch0={math.degrees(self.pitch0):.3f} deg, '
                f'x0={self.position_hold_reference:.3f} m'
            )

    def joint_states_ready(self) -> bool:
        return all(
            name in self.leg_position for name in self.LEG_JOINTS
        )

    def joint_states_fresh(self, now: float) -> bool:
        return (
            self.last_joint_state_wall_time is not None
            and now - self.last_joint_state_wall_time
            <= self.joint_state_timeout
        )

    def odom_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.pitch,
                self.pitch_rate,
                self.position_x,
                self.velocity_x,
                self.pitch0,
            )
        )

    def odom_fresh(self, now: float) -> bool:
        return (
            self.last_odom_wall_time is not None
            and now - self.last_odom_wall_time
            <= self.odom_timeout
        )

    # ------------------------------------------------------------------
    # Leg control
    # ------------------------------------------------------------------

    def base_leg_targets(self) -> List[float]:
        return [
            float(self.get_parameter('j1').value),
            float(self.get_parameter('j2').value),
            float(self.get_parameter('j3').value),
            float(self.get_parameter('j4').value),
        ]

    def desired_leg_targets(self) -> List[float]:
        lean_offset = 0.0
        if (
            self.drive_mode == 'leg_lean_assist'
            and self.forward_motion_active()
        ):
            lean_gain = float(self.get_parameter('leg_lean_gain').value)
            max_lean_offset = max(
                0.0,
                abs(float(self.get_parameter('max_leg_lean_offset').value)),
            )

            theta = 0.0
            if self.pitch is not None and self.pitch0 is not None:
                theta = self.pitch - self.pitch0

            pitch_error = self.last_pitch_ref - theta
            lean_offset = clamp(
                lean_gain * pitch_error,
                -max_lean_offset,
                max_lean_offset,
            )
            self.last_pitch_error = pitch_error
        else:
            self.last_pitch_error = 0.0

        lean_pattern = [
            float(self.get_parameter('leg_lean_j1').value),
            float(self.get_parameter('leg_lean_j2').value),
            float(self.get_parameter('leg_lean_j3').value),
            float(self.get_parameter('leg_lean_j4').value),
        ]
        self.last_leg_lean_offset = lean_offset

        return [
            base
            + extension_pattern * self.leg_extension_offset
            + lean_pattern_value * lean_offset
            for base, extension_pattern, lean_pattern_value in zip(
                self.base_leg_targets(),
                self.EXTENSION_PATTERN,
                lean_pattern,
            )
        ]

    def calculate_leg_lean_feedforward_efforts(self) -> List[float]:
        self.last_leg_lean_effort = 0.0
        if (
            self.drive_mode != 'leg_lean_assist'
            or not self.forward_motion_active()
        ):
            return [0.0, 0.0, 0.0, 0.0]

        deadband = math.radians(
            abs(float(self.get_parameter('pitch_error_deadband_deg').value))
        )
        pitch_error = self.last_pitch_error
        if abs(pitch_error) <= deadband:
            return [0.0, 0.0, 0.0, 0.0]

        active_error = pitch_error - math.copysign(deadband, pitch_error)
        effort_gain = float(
            self.get_parameter('leg_lean_effort_gain').value
        )
        effort_limit = max(
            0.0,
            abs(float(self.get_parameter('max_leg_lean_effort').value)),
        )
        effort = clamp(
            effort_gain * active_error,
            -effort_limit,
            effort_limit,
        )
        self.last_leg_lean_effort = effort

        lean_pattern = [
            float(self.get_parameter('leg_lean_j1').value),
            float(self.get_parameter('leg_lean_j2').value),
            float(self.get_parameter('leg_lean_j3').value),
            float(self.get_parameter('leg_lean_j4').value),
        ]
        return [
            sign_or_zero(pattern_value) * effort
            for pattern_value in lean_pattern
        ]

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

    def calculate_leg_efforts(
        self,
        now: float,
        dt: float,
    ) -> List[float]:
        if not self.joint_states_fresh(now):
            self.last_leg_lean_effort = 0.0
            return [0.0, 0.0, 0.0, 0.0]

        start_delay = max(
            0.0,
            float(self.get_parameter('start_delay').value),
        )
        if now - self.start_wall_time < start_delay:
            self.last_leg_lean_effort = 0.0
            return [0.0, 0.0, 0.0, 0.0]

        self.update_commanded_leg_targets(dt)
        if self.commanded_leg_targets is None:
            self.last_leg_lean_effort = 0.0
            return [0.0, 0.0, 0.0, 0.0]

        kp = float(self.get_parameter('leg_kp').value)
        kd = float(self.get_parameter('leg_kd').value)
        max_effort = max(
            0.0,
            abs(float(self.get_parameter('max_leg_effort').value)),
        )

        lean_feedforward = self.calculate_leg_lean_feedforward_efforts()

        efforts: List[float] = []
        for index, (name, target) in enumerate(zip(
            self.LEG_JOINTS,
            self.commanded_leg_targets,
        )):
            position = self.leg_position[name]
            velocity = self.leg_velocity.get(name, 0.0)

            effort = (
                kp * (target - position)
                - kd * velocity
                + lean_feedforward[index]
            )
            efforts.append(
                clamp(effort, -max_effort, max_effort)
            )

        return efforts

    # ------------------------------------------------------------------
    # Motion command and balance control
    # ------------------------------------------------------------------

    def forward_motion_active(self) -> bool:
        return self.motion_key in ('w', 's')

    def refresh_keyboard_command(self, now: float) -> None:
        if (
            self.motion_key is None
            or now - self.last_motion_key_wall_time
            > self.deadman_timeout
        ):
            self.motion_key = None
            self.requested_target_speed = 0.0
            self.requested_turn_effort = 0.0
            return

        speed = self.forward_sign * self.command_speed
        turn = self.turn_sign * self.turn_torque

        if self.motion_key == 'w':
            self.requested_target_speed = speed
            self.requested_turn_effort = 0.0
        elif self.motion_key == 's':
            self.requested_target_speed = -speed
            self.requested_turn_effort = 0.0
        elif self.motion_key == 'a':
            self.requested_target_speed = 0.0
            self.requested_turn_effort = turn
        elif self.motion_key == 'd':
            self.requested_target_speed = 0.0
            self.requested_turn_effort = -turn
        else:
            self.requested_target_speed = 0.0
            self.requested_turn_effort = 0.0

        self.requested_target_speed = clamp(
            self.requested_target_speed,
            -self.max_target_speed,
            self.max_target_speed,
        )
        self.requested_turn_effort = clamp(
            self.requested_turn_effort,
            -self.max_turn_torque,
            self.max_turn_torque,
        )

    def update_filtered_target_speed(self, dt: float) -> None:
        slew_rate = max(
            0.01,
            abs(
                float(
                    self.get_parameter(
                        'target_speed_slew_rate'
                    ).value
                )
            ),
        )
        max_step = slew_rate * max(0.0, dt)
        error = (
            self.requested_target_speed
            - self.filtered_target_speed
        )
        self.filtered_target_speed += clamp(
            error,
            -max_step,
            max_step,
        )

    def calculate_wheel_efforts(
        self,
        now: float,
        dt: float,
    ) -> Tuple[float, float]:
        self.refresh_keyboard_command(now)
        if self.forward_motion_active():
            self.update_filtered_target_speed(dt)
        else:
            self.filtered_target_speed = 0.0
            self.last_pitch_ref = 0.0

        if not self.odom_fresh(now) or not self.odom_ready():
            self.last_safety_reason = 'waiting for fresh /base_odom'
            return 0.0, 0.0

        assert self.pitch is not None
        assert self.pitch_rate is not None
        assert self.position_x is not None
        assert self.velocity_x is not None
        assert self.pitch0 is not None

        theta = self.pitch - self.pitch0
        theta_dot = self.pitch_rate
        velocity = self.velocity_x

        fall_angle = math.radians(
            abs(float(self.get_parameter('fall_angle_deg').value))
        )
        if abs(theta) > fall_angle:
            self.last_safety_reason = (
                f'fallen: theta={math.degrees(theta):.1f} deg'
            )
            self.filtered_target_speed = 0.0
            return 0.0, 0.0

        speed_error = 0.0
        if self.forward_motion_active():
            speed_error = self.filtered_target_speed - velocity

        max_pitch_ref = math.radians(
            abs(
                float(
                    self.get_parameter(
                        'max_pitch_ref_deg'
                    ).value
                )
            )
        )
        speed_to_pitch_gain = float(
            self.get_parameter('speed_to_pitch_gain').value
        )

        pitch_ref = self.pitch_ref_sign * clamp(
            speed_to_pitch_gain * speed_error,
            -max_pitch_ref,
            max_pitch_ref,
        )

        balance_kp = float(
            self.get_parameter('balance_kp').value
        )
        balance_kd = float(
            self.get_parameter('balance_kd').value
        )
        velocity_kd = float(
            self.get_parameter('velocity_kd').value
        )
        position_hold_kp = float(
            self.get_parameter('position_hold_kp').value
        )

        position_error = 0.0
        if (
            abs(self.filtered_target_speed) < 1e-3
            and self.position_hold_reference is not None
        ):
            position_error = (
                self.position_x - self.position_hold_reference
            )
        else:
            # While moving, continually move the stationary hold reference
            # along with the robot to avoid pulling it back to the start.
            self.position_hold_reference = self.position_x

        if self.drive_mode == 'leg_lean_assist':
            assist_pitch_kp = float(
                self.get_parameter('wheel_assist_pitch_kp').value
            )
            assist_pitch_kd = float(
                self.get_parameter('wheel_assist_pitch_kd').value
            )
            assist_velocity_kd = float(
                self.get_parameter('wheel_assist_velocity_kd').value
            )
            assist_limit = max(
                0.0,
                abs(
                    float(
                        self.get_parameter(
                            'max_wheel_assist_torque'
                        ).value
                    )
                ),
            )

            if self.forward_motion_active():
                raw_drive_effort = self.balance_sign * (
                    assist_pitch_kp * (theta - pitch_ref)
                    + assist_pitch_kd * theta_dot
                    + assist_velocity_kd
                    * (velocity - self.filtered_target_speed)
                )
            else:
                raw_drive_effort = 0.0
            raw_drive_effort = clamp(
                raw_drive_effort,
                -assist_limit,
                assist_limit,
            )
        else:
            raw_drive_effort = self.balance_sign * (
                balance_kp * (theta - pitch_ref)
                + balance_kd * theta_dot
                + velocity_kd * (velocity - self.filtered_target_speed)
                + position_hold_kp * position_error
            )

        turn_effort = self.requested_turn_effort
        wheel1 = raw_drive_effort + turn_effort
        wheel2 = raw_drive_effort - turn_effort

        wheel1 *= self.wheel1_sign
        wheel2 *= self.wheel2_sign

        wheel1 = clamp(
            wheel1,
            -self.max_wheel_torque,
            self.max_wheel_torque,
        )
        wheel2 = clamp(
            wheel2,
            -self.max_wheel_torque,
            self.max_wheel_torque,
        )

        self.last_pitch_ref = pitch_ref
        self.last_drive_effort = raw_drive_effort
        self.last_safety_reason = 'active'
        return wheel1, wheel2

    # ------------------------------------------------------------------
    # Publishing and loop
    # ------------------------------------------------------------------

    def publish_wheels(
        self,
        values: Tuple[float, float],
    ) -> None:
        msg = Float64MultiArray()
        msg.data = [float(values[0]), float(values[1])]
        self.wheel_pub.publish(msg)
        self.last_wheel_command = values

    def publish_legs(self, values: List[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(value) for value in values]
        self.leg_pub.publish(msg)
        self.last_leg_command = values

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
            self.last_safety_reason = 'stale /joint_states'
            self.publish_wheels((0.0, 0.0))
            self.publish_legs([0.0, 0.0, 0.0, 0.0])
            self.print_runtime_status(now)
            return

        wheel_efforts = self.calculate_wheel_efforts(now, dt)
        leg_efforts = self.calculate_leg_efforts(now, dt)

        self.publish_legs(leg_efforts)
        self.publish_wheels(wheel_efforts)
        self.print_runtime_status(now)

    def print_runtime_status(self, now: float) -> None:
        status_period = max(
            0.05,
            float(self.get_parameter('status_period').value),
        )
        if now - self.last_status_wall_time < status_period:
            return
        self.last_status_wall_time = now

        theta_deg = float('nan')
        pitch_ref_deg = math.degrees(self.last_pitch_ref)
        velocity = float('nan')

        if self.pitch is not None and self.pitch0 is not None:
            theta_deg = math.degrees(self.pitch - self.pitch0)
        if self.velocity_x is not None:
            velocity = self.velocity_x

        print(
            '\r'
            f'mode={(self.motion_key or "STOP").upper():>4s} | '
            f'v_ref={self.filtered_target_speed:+.3f} m/s | '
            f'v={velocity:+.3f} m/s | '
            f'theta={theta_deg:+.2f} deg | '
            f'theta_ref={pitch_ref_deg:+.2f} deg | '
            f'theta_err={math.degrees(self.last_pitch_error):+.2f} deg | '
            f'lean={self.last_leg_lean_offset:+.3f} rad | '
            f'lean_ff={self.last_leg_lean_effort:+.3f} N*m | '
            f'wheel=[{self.last_wheel_command[0]:+.3f}, '
            f'{self.last_wheel_command[1]:+.3f}] N*m | '
            f'{self.last_safety_reason}      ',
            end='',
            flush=True,
        )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def stop_motion(self) -> None:
        self.motion_key = None
        self.requested_target_speed = 0.0
        self.filtered_target_speed = 0.0
        self.requested_turn_effort = 0.0
        self.last_pitch_ref = 0.0
        self.last_pitch_error = 0.0
        self.last_leg_lean_offset = 0.0
        self.last_leg_lean_effort = 0.0

    def adjust_leg_extension(self, direction: float) -> None:
        self.leg_extension_offset = clamp(
            self.leg_extension_offset
            + direction * self.leg_extension_step,
            -self.max_leg_extension_offset,
            self.max_leg_extension_offset,
        )
        self.print_configuration()

    def handle_key(self, key: str) -> None:
        key = key.lower()

        if key in ('w', 'a', 's', 'd'):
            self.motion_key = key
            self.last_motion_key_wall_time = time.monotonic()
            return

        if key == ' ':
            self.stop_motion()
            print('\nMotion command stopped; leg PD remains active.')
            return

        if key == 'r':
            self.adjust_leg_extension(+1.0)
            return

        if key == 'f':
            self.adjust_leg_extension(-1.0)
            return

        if key == '0':
            self.leg_extension_offset = 0.0
            self.print_configuration()
            return

        if key == ']':
            self.command_speed = min(
                self.max_target_speed,
                self.command_speed + self.speed_step,
            )
            self.print_configuration()
            return

        if key == '[':
            self.command_speed = max(
                0.0,
                self.command_speed - self.speed_step,
            )
            self.print_configuration()
            return

        if key == '=':
            self.turn_torque = min(
                self.max_turn_torque,
                self.turn_torque + self.turn_torque_step,
            )
            self.print_configuration()
            return

        if key == '-':
            self.turn_torque = max(
                0.0,
                self.turn_torque - self.turn_torque_step,
            )
            self.print_configuration()
            return

        if key == 'p':
            self.print_configuration()
            return

        if key == 'h':
            print('\n' + HELP)
            self.print_configuration()
            return

        if key == 'q' or key == '\x03':
            self.quit_requested = True

    def print_configuration(self) -> None:
        targets = self.desired_leg_targets()
        print(
            '\nConfiguration: '
            f'command_speed={self.command_speed:.3f} m/s, '
            f'turn={self.turn_torque:.3f} N*m, '
            f'drive_mode={self.drive_mode}, '
            f'leg_offset={self.leg_extension_offset:+.3f} rad, '
            f'lean_offset={self.last_leg_lean_offset:+.3f} rad, '
            'leg_targets=['
            + ', '.join(f'{value:+.3f}' for value in targets)
            + '], '
            f'balance_sign={self.balance_sign:+.0f}, '
            f'pitch_ref_sign={self.pitch_ref_sign:+.0f}, '
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
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                old_settings,
            )
            print()



PACKAGE_NAME = 'wheel_leg_description'
DEFAULT_PARAMS_FILENAME = 'six_motor_manual_controller.yaml'


def find_default_params_file() -> Path:
    """Locate the controller YAML, preferring the source workspace copy.

    Search order:
      1. Source package beside this script (normal symlink-install).
      2. Source workspace inferred from the installed package prefix.
      3. ROS 2 package share directory.
      4. Install prefix inferred directly from this script.

    Preferring the source YAML means the usual development workflow is:
      edit src/wheel_leg_description/config/*.yaml
      ros2 run wheel_leg_description six_motor_manual_controller.py

    No rebuild is needed for parameter-only changes when the source file can
    be located. The installed share copy remains a safe fallback.

    Raises:
        FileNotFoundError: if no usable YAML file can be located.
    """
    candidates: List[Path] = []
    resolved_script = Path(__file__).resolve()

    # Source layout:
    #   wheel_leg_description/scripts/controller.py
    #   wheel_leg_description/config/controller.yaml
    if resolved_script.parent.name == 'scripts':
        candidates.append(
            resolved_script.parent.parent
            / 'config'
            / DEFAULT_PARAMS_FILENAME
        )

    try:
        share_directory = Path(
            get_package_share_directory(PACKAGE_NAME)
        ).resolve()

        # Typical colcon layout:
        #   <workspace>/install/<package>/share/<package>
        #   <workspace>/src/<package>/config
        try:
            workspace_root = share_directory.parents[3]
            candidates.append(
                workspace_root
                / 'src'
                / PACKAGE_NAME
                / 'config'
                / DEFAULT_PARAMS_FILENAME
            )
        except IndexError:
            pass

        candidates.append(
            share_directory
            / 'config'
            / DEFAULT_PARAMS_FILENAME
        )
    except PackageNotFoundError:
        pass

    # Installed layout:
    #   <prefix>/lib/wheel_leg_description/controller.py
    #   <prefix>/share/wheel_leg_description/config/controller.yaml
    if (
        resolved_script.parent.name == PACKAGE_NAME
        and resolved_script.parent.parent.name == 'lib'
    ):
        install_prefix = resolved_script.parent.parent.parent
        candidates.append(
            install_prefix
            / 'share'
            / PACKAGE_NAME
            / 'config'
            / DEFAULT_PARAMS_FILENAME
        )

    checked: List[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in checked:
            continue
        checked.append(candidate)

        if candidate.is_file():
            return candidate

    checked_text = '\n'.join(
        f'  - {candidate}' for candidate in checked
    )
    raise FileNotFoundError(
        'Could not locate the default parameter file '
        f'{DEFAULT_PARAMS_FILENAME}.\n'
        'Checked:\n'
        f'{checked_text or "  - no candidate paths available"}'
    )


def contains_params_file(arguments: Sequence[str]) -> bool:
    """Return True when the user already supplied a params file."""
    return any(
        argument == '--params-file'
        or argument.startswith('--params-file=')
        for argument in arguments
    )


def add_default_params_file(
    arguments: Sequence[str],
    params_file: Path,
) -> List[str]:
    """Add the default YAML before user -p overrides.

    When the user already provides --params-file, their file is used
    unchanged. Otherwise the package YAML is inserted immediately after
    --ros-args so any later command-line -p values retain higher priority.
    """
    effective_arguments = list(arguments)

    if contains_params_file(effective_arguments):
        return effective_arguments

    ros_args = ['--params-file', str(params_file)]

    if '--ros-args' in effective_arguments:
        insert_index = effective_arguments.index('--ros-args') + 1
        effective_arguments[insert_index:insert_index] = ros_args
    else:
        effective_arguments.extend(['--ros-args', *ros_args])

    return effective_arguments

def main(args=None) -> None:
    # Automatically load the package YAML when ros2 run is used without
    # an explicit --params-file. Explicit command-line -p overrides remain
    # higher priority because the YAML is inserted before user ROS arguments.
    original_arguments = (
        list(sys.argv)
        if args is None
        else list(args)
    )

    using_explicit_params_file = contains_params_file(
        original_arguments
    )

    if using_explicit_params_file:
        effective_arguments = original_arguments
        print(
            '[PARAMS] Using the --params-file supplied '
            'on the command line.'
        )
    else:
        default_params_file = find_default_params_file()
        effective_arguments = add_default_params_file(
            original_arguments,
            default_params_file,
        )
        print(
            '[PARAMS] Automatically loading:\n'
            f'  {default_params_file}'
        )

    # Keep the ROS context alive after Ctrl-C until final zero commands
    # have been published.
    rclpy.init(
        args=effective_arguments,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = SixMotorBalanceTeleop()

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
                print(
                    'Warning: failed to publish final zero effort: '
                    f'{exc}'
                )

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
