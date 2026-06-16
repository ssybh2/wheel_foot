#!/usr/bin/env python3
"""One-shot wheel-height calibration for the wheel-leg robot.

This node is only a calibration tool.

It:
1. Ramps the four active leg joints to a nominal symmetric configuration.
2. Reads the two wheel-link world Z coordinates from /gazebo/link_states.
3. Automatically probes the sign of the differential leg correction.
4. Adjusts a single differential offset until both wheel centers have
   nearly identical heights.
5. Compares wheel-center relative heights in the base_link frame.
6. Saves the final four joint targets to a ROS 2 parameter YAML file.
7. Holds the calibrated targets until the node is stopped.

Do not run this node together with leg_effort_symmetric_hold.py.
"""

from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import rclpy
from gazebo_msgs.msg import LinkStates
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class LegHeightCalibrator(Node):
    JOINTS = [
        'big_arm_1_joint',
        'big_arm_2_joint',
        'big_arm_3_joint',
        'big_arm_4_joint',
    ]

    BASE_LINK = 'base_link'
    WHEEL_1_LINK = 'wheel_motor_1_link'
    WHEEL_2_LINK = 'wheel_motor_2_link'

    def __init__(self) -> None:
        super().__init__('leg_height_calibrator')

        # Nominal symmetric leg configuration.
        self.declare_parameter('extension_angle', 0.15)

        # Inner joint PD controller.
        self.declare_parameter('kp', 4.0)
        self.declare_parameter('kd', 0.5)
        self.declare_parameter('max_effort', 2.0)

        # Startup and smooth target motion.
        self.declare_parameter('start_delay', 0.5)
        self.declare_parameter('ramp_time', 4.0)
        self.declare_parameter('base_settle_time', 1.5)

        # Automatic sign probe.
        self.declare_parameter('probe_offset', 0.006)
        self.declare_parameter('probe_time', 1.5)
        self.declare_parameter('min_probe_response_m', 0.00002)

        # Differential-height calibration.
        # The update law is:
        # delta_dot = correction_sign * level_gain * height_error
        # level_gain units are approximately rad / (m*s).
        self.declare_parameter('level_gain', 3.0)
        self.declare_parameter('max_delta', 0.04)

        # Height filtering and convergence.
        self.declare_parameter('height_filter_alpha', 0.90)
        self.declare_parameter('height_tolerance_m', 0.00020)
        self.declare_parameter('settle_time', 2.0)
        self.declare_parameter('timeout', 40.0)

        self.declare_parameter(
            'save_file',
            '/home/hby/foot_ws/src/wheel_leg_description/config/'
            'leg_calibration.yaml',
        )

        self.position: Dict[str, float] = {}
        self.velocity: Dict[str, float] = {}

        self.wheel_1_z_base: Optional[float] = None
        self.wheel_2_z_base: Optional[float] = None
        self.height_error: Optional[float] = None
        self.filtered_height_error: Optional[float] = None

        self.state = 'WAIT'
        self.state_start = self.now_sec()
        self.node_start = self.state_start
        self.last_loop_time = self.state_start

        self.initial_targets: Optional[List[float]] = None

        self.delta = 0.0
        self.final_delta = 0.0
        self.correction_sign: Optional[float] = None

        self.plus_samples: List[float] = []
        self.minus_samples: List[float] = []

        self.level_start: Optional[float] = None
        self.good_since: Optional[float] = None

        self.best_abs_error = float('inf')
        self.best_delta = 0.0
        self.saved = False

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            20,
        )

        self.create_subscription(
            LinkStates,
            '/gazebo/link_states',
            self.link_states_callback,
            20,
        )

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            '/leg_effort_controller/commands',
            10,
        )

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('leg_height_calibrator started.')
        self.get_logger().warning(
            'Do not run leg_effort_symmetric_hold.py at the same time.'
        )
        self.get_logger().info(
            'Calibration feedback: wheel relative Z difference in base_link.'
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def joint_state_callback(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if name not in self.JOINTS:
                continue

            if index < len(msg.position):
                self.position[name] = float(msg.position[index])

            if index < len(msg.velocity):
                self.velocity[name] = float(msg.velocity[index])

    @staticmethod
    def find_link_index(names: List[str], short_name: str) -> Optional[int]:
        for index, full_name in enumerate(names):
            if full_name == short_name or full_name.endswith(
                f'::{short_name}'
            ):
                return index
        return None

    @staticmethod
    def world_point_to_base_local_z(base_pose, point_pose) -> float:
        # The base quaternion maps base-frame vectors into the world frame.
        # Therefore local = R^T * (point_world - base_world).
        q = base_pose.orientation
        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)

        norm = (x * x + y * y + z * z + w * w) ** 0.5
        if norm <= 1.0e-12:
            return 0.0

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        dx = float(point_pose.position.x - base_pose.position.x)
        dy = float(point_pose.position.y - base_pose.position.y)
        dz = float(point_pose.position.z - base_pose.position.z)

        # Third component of R^T * [dx, dy, dz].
        r02 = 2.0 * (x * z + y * w)
        r12 = 2.0 * (y * z - x * w)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        return r02 * dx + r12 * dy + r22 * dz

    def link_states_callback(self, msg: LinkStates) -> None:
        base_index = self.find_link_index(msg.name, self.BASE_LINK)
        index_1 = self.find_link_index(msg.name, self.WHEEL_1_LINK)
        index_2 = self.find_link_index(msg.name, self.WHEEL_2_LINK)

        if base_index is None or index_1 is None or index_2 is None:
            return

        if (
            base_index >= len(msg.pose)
            or index_1 >= len(msg.pose)
            or index_2 >= len(msg.pose)
        ):
            return

        base_pose = msg.pose[base_index]

        self.wheel_1_z_base = self.world_point_to_base_local_z(
            base_pose,
            msg.pose[index_1],
        )
        self.wheel_2_z_base = self.world_point_to_base_local_z(
            base_pose,
            msg.pose[index_2],
        )

        # Negative values normally mean that a wheel is below base_link.
        # The difference is the left/right leg-height mismatch relative
        # to the robot body, independent of world ground height and roll.
        measured_error = (
            self.wheel_1_z_base
            - self.wheel_2_z_base
        )
        self.height_error = measured_error

        alpha = clamp(
            float(self.get_parameter('height_filter_alpha').value),
            0.0,
            0.9999,
        )

        if self.filtered_height_error is None:
            self.filtered_height_error = measured_error
        else:
            self.filtered_height_error = (
                alpha * self.filtered_height_error
                + (1.0 - alpha) * measured_error
            )

    def ready(self) -> bool:
        return (
            self.filtered_height_error is not None
            and all(name in self.position for name in self.JOINTS)
        )

    def nominal_targets(self) -> List[float]:
        a = float(self.get_parameter('extension_angle').value)
        return [a, -a, a, -a]

    def targets_from_delta(self, delta: float) -> List[float]:
        """Generate four joint targets from one differential correction.

        wheel_1 side uses big_arm_2 and big_arm_3.
        wheel_2 side uses big_arm_1 and big_arm_4.

        Positive delta:
          wheel_1 extension = a + delta
          wheel_2 extension = a - delta
        """
        a = float(self.get_parameter('extension_angle').value)

        wheel_1_extension = a + delta
        wheel_2_extension = a - delta

        return [
            +wheel_2_extension,  # big_arm_1_joint
            -wheel_1_extension,  # big_arm_2_joint
            +wheel_1_extension,  # big_arm_3_joint
            -wheel_2_extension,  # big_arm_4_joint
        ]

    def publish_pd(self, targets: List[float]) -> None:
        kp = float(self.get_parameter('kp').value)
        kd = float(self.get_parameter('kd').value)
        max_effort = abs(
            float(self.get_parameter('max_effort').value)
        )

        efforts: List[float] = []

        for name, target in zip(self.JOINTS, targets):
            position = self.position.get(name, 0.0)
            velocity = self.velocity.get(name, 0.0)

            effort = (
                kp * (target - position)
                - kd * velocity
            )
            effort = clamp(
                effort,
                -max_effort,
                max_effort,
            )
            efforts.append(effort)

        msg = Float64MultiArray()
        msg.data = efforts
        self.command_pub.publish(msg)

    def transition(self, new_state: str) -> None:
        self.state = new_state
        self.state_start = self.now_sec()
        self.get_logger().info(
            f'Calibration state: {new_state}'
        )

    def collect_probe_sample(self, storage: List[float]) -> None:
        probe_time = float(
            self.get_parameter('probe_time').value
        )
        elapsed = self.now_sec() - self.state_start

        # Ignore the first half of the probe motion.
        if (
            elapsed >= 0.5 * probe_time
            and self.filtered_height_error is not None
        ):
            storage.append(self.filtered_height_error)

    def finish_probe(self) -> None:
        if not self.plus_samples or not self.minus_samples:
            self.get_logger().error(
                'Probe data missing.'
            )
            self.transition('ERROR')
            return

        error_plus = mean(self.plus_samples)
        error_minus = mean(self.minus_samples)
        response = error_plus - error_minus

        minimum_response = abs(
            float(
                self.get_parameter(
                    'min_probe_response_m'
                ).value
            )
        )

        self.get_logger().info(
            'Probe result: '
            f'error(+delta)={error_plus * 1000.0:+.6f} mm, '
            f'error(-delta)={error_minus * 1000.0:+.6f} mm, '
            f'response={response * 1000.0:+.6f} mm'
        )

        if abs(response) < minimum_response:
            self.get_logger().error(
                'Probe response is too small. '
                'Increase probe_offset or probe_time.'
            )
            self.transition('ERROR')
            return

        # If positive delta increases the height error, a positive error
        # must cause delta to decrease.
        self.correction_sign = (
            -1.0 if response > 0.0 else 1.0
        )

        self.delta = 0.0
        self.level_start = self.now_sec()
        self.good_since = None
        self.best_abs_error = float('inf')
        self.best_delta = 0.0

        self.get_logger().info(
            'Automatic correction sign identified: '
            f'{self.correction_sign:+.0f}'
        )
        self.transition('LEVEL')

    def save_result(self, converged: bool) -> None:
        if self.saved:
            return

        targets = self.targets_from_delta(
            self.final_delta
        )

        extension = float(
            self.get_parameter('extension_angle').value
        )

        save_path = Path(
            str(
                self.get_parameter(
                    'save_file'
                ).value
            )
        ).expanduser()

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        error_mm = (
            self.filtered_height_error * 1000.0
            if self.filtered_height_error is not None
            else float('nan')
        )

        content = (
            '# Generated by leg_height_calibrator.py\n'
            f'# converged: {str(converged).lower()}\n'
            f'# base_extension_angle_rad: {extension:.9f}\n'
            f'# differential_delta_rad: '
            f'{self.final_delta:+.9f}\n'
            f'# wheel_1_extension_rad: '
            f'{extension + self.final_delta:.9f}\n'
            f'# wheel_2_extension_rad: '
            f'{extension - self.final_delta:.9f}\n'
            f'# final_wheel_height_error_mm: '
            f'{error_mm:+.9f}\n'
            'leg_effort_symmetric_hold:\n'
            '  ros__parameters:\n'
            f'    j1: {targets[0]:.9f}\n'
            f'    j2: {targets[1]:.9f}\n'
            f'    j3: {targets[2]:.9f}\n'
            f'    j4: {targets[3]:.9f}\n'
            f'    kp: '
            f'{float(self.get_parameter("kp").value):.9f}\n'
            f'    kd: '
            f'{float(self.get_parameter("kd").value):.9f}\n'
            f'    max_effort: '
            f'{float(self.get_parameter("max_effort").value):.9f}\n'
            '    start_delay: 0.5\n'
            '    use_sim_time: true\n'
        )

        save_path.write_text(
            content,
            encoding='utf-8',
        )

        self.saved = True

        self.get_logger().info('=' * 64)
        self.get_logger().info(
            'LEG HEIGHT CALIBRATION COMPLETE'
        )
        self.get_logger().info(
            f'converged: {converged}'
        )
        self.get_logger().info(
            f'delta = {self.final_delta:+.9f} rad'
        )
        self.get_logger().info(
            'targets: '
            f'j1={targets[0]:+.9f}, '
            f'j2={targets[1]:+.9f}, '
            f'j3={targets[2]:+.9f}, '
            f'j4={targets[3]:+.9f}'
        )
        self.get_logger().info(
            'final wheel height error = '
            f'{error_mm:+.6f} mm'
        )
        self.get_logger().info(
            f'saved to: {save_path}'
        )
        self.get_logger().info('=' * 64)

    def control_loop(self) -> None:
        now = self.now_sec()
        dt = clamp(
            now - self.last_loop_time,
            0.0,
            0.05,
        )
        self.last_loop_time = now

        if not self.ready():
            return

        start_delay = float(
            self.get_parameter('start_delay').value
        )

        if now - self.node_start < start_delay:
            return

        if self.state == 'WAIT':
            self.initial_targets = [
                self.position[name]
                for name in self.JOINTS
            ]
            self.transition('RAMP')

        if self.state == 'RAMP':
            assert self.initial_targets is not None

            ramp_time = max(
                0.0,
                float(
                    self.get_parameter(
                        'ramp_time'
                    ).value
                ),
            )

            if ramp_time <= 0.0:
                alpha = 1.0
            else:
                alpha = clamp(
                    (now - self.state_start)
                    / ramp_time,
                    0.0,
                    1.0,
                )

            nominal = self.nominal_targets()

            targets = [
                initial
                + alpha * (target - initial)
                for initial, target in zip(
                    self.initial_targets,
                    nominal,
                )
            ]

            self.publish_pd(targets)

            if alpha >= 1.0:
                self.transition('BASE_SETTLE')
            return

        if self.state == 'BASE_SETTLE':
            self.publish_pd(
                self.nominal_targets()
            )

            settle = float(
                self.get_parameter(
                    'base_settle_time'
                ).value
            )

            if now - self.state_start >= settle:
                self.plus_samples.clear()
                self.transition('PROBE_PLUS')
            return

        if self.state == 'PROBE_PLUS':
            probe = abs(
                float(
                    self.get_parameter(
                        'probe_offset'
                    ).value
                )
            )

            self.publish_pd(
                self.targets_from_delta(+probe)
            )
            self.collect_probe_sample(
                self.plus_samples
            )

            probe_time = float(
                self.get_parameter(
                    'probe_time'
                ).value
            )

            if now - self.state_start >= probe_time:
                self.minus_samples.clear()
                self.transition('PROBE_MINUS')
            return

        if self.state == 'PROBE_MINUS':
            probe = abs(
                float(
                    self.get_parameter(
                        'probe_offset'
                    ).value
                )
            )

            self.publish_pd(
                self.targets_from_delta(-probe)
            )
            self.collect_probe_sample(
                self.minus_samples
            )

            probe_time = float(
                self.get_parameter(
                    'probe_time'
                ).value
            )

            if now - self.state_start >= probe_time:
                self.finish_probe()
            return

        if self.state == 'LEVEL':
            assert self.filtered_height_error is not None
            assert self.correction_sign is not None
            assert self.level_start is not None

            gain = float(
                self.get_parameter(
                    'level_gain'
                ).value
            )
            max_delta = abs(
                float(
                    self.get_parameter(
                        'max_delta'
                    ).value
                )
            )

            delta_rate = (
                self.correction_sign
                * gain
                * self.filtered_height_error
            )

            self.delta += delta_rate * dt
            self.delta = clamp(
                self.delta,
                -max_delta,
                max_delta,
            )

            self.publish_pd(
                self.targets_from_delta(
                    self.delta
                )
            )

            abs_error = abs(
                self.filtered_height_error
            )

            if abs_error < self.best_abs_error:
                self.best_abs_error = abs_error
                self.best_delta = self.delta

            tolerance = abs(
                float(
                    self.get_parameter(
                        'height_tolerance_m'
                    ).value
                )
            )

            if abs_error <= tolerance:
                if self.good_since is None:
                    self.good_since = now
            else:
                self.good_since = None

            settle_time = float(
                self.get_parameter(
                    'settle_time'
                ).value
            )

            if (
                self.good_since is not None
                and now - self.good_since
                >= settle_time
            ):
                self.final_delta = self.delta
                self.save_result(
                    converged=True
                )
                self.transition('DONE')
                return

            timeout = float(
                self.get_parameter(
                    'timeout'
                ).value
            )

            if (
                now - self.level_start
                >= timeout
            ):
                self.final_delta = self.best_delta
                self.get_logger().warning(
                    'Timeout reached; saving the '
                    'best observed offset.'
                )
                self.save_result(
                    converged=False
                )
                self.transition('DONE')
            return

        if self.state == 'DONE':
            self.publish_pd(
                self.targets_from_delta(
                    self.final_delta
                )
            )
            return

        if self.state == 'ERROR':
            self.publish_pd(
                self.nominal_targets()
            )

    def stop(self) -> None:
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self.command_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LegHeightCalibrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
