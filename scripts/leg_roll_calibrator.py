#!/usr/bin/env python3
import math
from pathlib import Path
from statistics import mean

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def clamp(value, low, high):
    return max(low, min(high, value))


def quat_to_roll(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(sinr_cosp, cosr_cosp)


class LegRollCalibrator(Node):
    JOINTS = [
        'big_arm_1_joint',
        'big_arm_2_joint',
        'big_arm_3_joint',
        'big_arm_4_joint',
    ]

    def __init__(self):
        super().__init__('leg_roll_calibrator')

        self.declare_parameter('extension_angle', 0.15)

        self.declare_parameter('kp', 4.0)
        self.declare_parameter('kd', 0.5)
        self.declare_parameter('max_effort', 2.0)

        self.declare_parameter('start_delay', 0.5)
        self.declare_parameter('ramp_time', 4.0)
        self.declare_parameter('base_settle_time', 1.5)

        self.declare_parameter('probe_offset', 0.00004)
        self.declare_parameter('probe_time', 1.2)
        self.declare_parameter('min_probe_response_deg', 0.03)

        self.declare_parameter('level_gain', 0.18)
        self.declare_parameter('level_damping', 0.015)
        self.declare_parameter('max_delta', 0.04)

        self.declare_parameter('filter_alpha', 0.95)
        self.declare_parameter('roll_tolerance_deg', 0.15)
        self.declare_parameter('roll_rate_tolerance_deg_s', 0.25)
        self.declare_parameter('settle_time', 2.0)
        self.declare_parameter('timeout', 30.0)

        self.declare_parameter(
            'save_file',
            '/home/hby/foot_ws/src/wheel_leg_description/config/leg_calibration.yaml'
        )

        self.pos = {}
        self.vel = {}

        self.roll = None
        self.roll_f = None
        self.roll_rate = 0.0

        self.state = 'WAIT'
        self.state_start = self.now_sec()
        self.node_start = self.state_start
        self.last_time = self.state_start

        self.initial_targets = None
        self.delta = 0.0
        self.final_delta = 0.0
        self.control_sign = None

        self.plus_samples = []
        self.minus_samples = []

        self.level_start = None
        self.good_since = None
        self.best_abs_roll = float('inf')
        self.best_delta = 0.0
        self.saved = False

        self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 20
        )
        self.create_subscription(
            Odometry, '/base_odom', self.odom_cb, 20
        )
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/leg_effort_controller/commands',
            10
        )

        self.timer = self.create_timer(0.01, self.loop)

        self.get_logger().info('leg_roll_calibrator started')
        self.get_logger().warning(
            'Do not run leg_effort_symmetric_hold.py at the same time.'
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name not in self.JOINTS:
                continue
            if i < len(msg.position):
                self.pos[name] = float(msg.position[i])
            if i < len(msg.velocity):
                self.vel[name] = float(msg.velocity[i])

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.roll = quat_to_roll(q.x, q.y, q.z, q.w)
        self.roll_rate = float(msg.twist.twist.angular.x)

        alpha = clamp(
            float(self.get_parameter('filter_alpha').value),
            0.0,
            0.9999
        )

        if self.roll_f is None:
            self.roll_f = self.roll
        else:
            self.roll_f = (
                alpha * self.roll_f
                + (1.0 - alpha) * self.roll
            )

    def ready(self):
        return (
            self.roll_f is not None
            and all(name in self.pos for name in self.JOINTS)
        )

    def nominal_targets(self):
        a = float(self.get_parameter('extension_angle').value)
        return [a, -a, a, -a]

    def targets_from_delta(self, delta):
        a = float(self.get_parameter('extension_angle').value)

        wheel_1_extension = a + delta
        wheel_2_extension = a - delta

        return [
            +wheel_2_extension,
            -wheel_1_extension,
            +wheel_1_extension,
            -wheel_2_extension,
        ]

    def publish_pd(self, targets):
        kp = float(self.get_parameter('kp').value)
        kd = float(self.get_parameter('kd').value)
        max_effort = abs(
            float(self.get_parameter('max_effort').value)
        )

        efforts = []

        for name, target in zip(self.JOINTS, targets):
            p = self.pos.get(name, 0.0)
            v = self.vel.get(name, 0.0)

            effort = kp * (target - p) - kd * v
            effort = clamp(effort, -max_effort, max_effort)
            efforts.append(effort)

        msg = Float64MultiArray()
        msg.data = efforts
        self.pub.publish(msg)

    def change_state(self, state):
        self.state = state
        self.state_start = self.now_sec()
        self.get_logger().info(f'Calibration state: {state}')

    def collect_probe(self, storage):
        probe_time = float(self.get_parameter('probe_time').value)
        elapsed = self.now_sec() - self.state_start

        if elapsed >= 0.5 * probe_time and self.roll_f is not None:
            storage.append(self.roll_f)

    def finish_probe(self):
        if not self.plus_samples or not self.minus_samples:
            self.get_logger().error('Probe data missing.')
            self.change_state('ERROR')
            return

        roll_plus = mean(self.plus_samples)
        roll_minus = mean(self.minus_samples)
        response = roll_plus - roll_minus

        min_response = math.radians(
            float(
                self.get_parameter(
                    'min_probe_response_deg'
                ).value
            )
        )

        self.get_logger().info(
            'Probe: '
            f'roll(+delta)={math.degrees(roll_plus):+.4f} deg, '
            f'roll(-delta)={math.degrees(roll_minus):+.4f} deg'
        )

        if abs(response) < min_response:
            self.get_logger().error(
                'Probe response too small. Increase probe_offset slightly.'
            )
            self.change_state('ERROR')
            return

        self.control_sign = -1.0 if response > 0.0 else 1.0

        self.delta = 0.0
        self.level_start = self.now_sec()
        self.good_since = None
        self.best_abs_roll = float('inf')
        self.best_delta = 0.0

        self.get_logger().info(
            f'Automatic correction sign = {self.control_sign:+.0f}'
        )
        self.change_state('LEVEL')

    def save_result(self, converged):
        if self.saved:
            return

        targets = self.targets_from_delta(self.final_delta)
        extension = float(
            self.get_parameter('extension_angle').value
        )

        save_path = Path(
            str(self.get_parameter('save_file').value)
        ).expanduser()

        save_path.parent.mkdir(parents=True, exist_ok=True)

        roll_deg = (
            math.degrees(self.roll_f)
            if self.roll_f is not None
            else float('nan')
        )

        content = (
            '# Generated by leg_roll_calibrator.py\n'
            f'# converged: {str(converged).lower()}\n'
            f'# base_extension_angle_rad: {extension:.9f}\n'
            f'# differential_delta_rad: {self.final_delta:+.9f}\n'
            f'# wheel_1_extension_rad: '
            f'{extension + self.final_delta:.9f}\n'
            f'# wheel_2_extension_rad: '
            f'{extension - self.final_delta:.9f}\n'
            f'# final_roll_deg: {roll_deg:+.6f}\n'
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

        save_path.write_text(content, encoding='utf-8')
        self.saved = True

        self.get_logger().info('=' * 60)
        self.get_logger().info('LEG CALIBRATION COMPLETE')
        self.get_logger().info(f'converged: {converged}')
        self.get_logger().info(
            f'delta = {self.final_delta:+.6f} rad'
        )
        self.get_logger().info(
            'targets: '
            f'j1={targets[0]:+.6f}, '
            f'j2={targets[1]:+.6f}, '
            f'j3={targets[2]:+.6f}, '
            f'j4={targets[3]:+.6f}'
        )
        self.get_logger().info(
            f'final roll = {roll_deg:+.4f} deg'
        )
        self.get_logger().info(f'saved to: {save_path}')
        self.get_logger().info('=' * 60)

    def loop(self):
        now = self.now_sec()
        dt = clamp(now - self.last_time, 0.0, 0.05)
        self.last_time = now

        if not self.ready():
            return

        start_delay = float(
            self.get_parameter('start_delay').value
        )

        if now - self.node_start < start_delay:
            return

        if self.state == 'WAIT':
            self.initial_targets = [
                self.pos[name] for name in self.JOINTS
            ]
            self.change_state('RAMP')

        if self.state == 'RAMP':
            ramp_time = max(
                0.0,
                float(self.get_parameter('ramp_time').value)
            )

            alpha = (
                1.0
                if ramp_time <= 0.0
                else clamp(
                    (now - self.state_start) / ramp_time,
                    0.0,
                    1.0
                )
            )

            nominal = self.nominal_targets()
            targets = [
                q0 + alpha * (q1 - q0)
                for q0, q1 in zip(
                    self.initial_targets,
                    nominal
                )
            ]

            self.publish_pd(targets)

            if alpha >= 1.0:
                self.change_state('BASE_SETTLE')
            return

        if self.state == 'BASE_SETTLE':
            self.publish_pd(self.nominal_targets())

            settle = float(
                self.get_parameter('base_settle_time').value
            )

            if now - self.state_start >= settle:
                self.plus_samples.clear()
                self.change_state('PROBE_PLUS')
            return

        if self.state == 'PROBE_PLUS':
            probe = abs(
                float(self.get_parameter('probe_offset').value)
            )

            self.publish_pd(
                self.targets_from_delta(+probe)
            )
            self.collect_probe(self.plus_samples)

            if now - self.state_start >= float(
                self.get_parameter('probe_time').value
            ):
                self.minus_samples.clear()
                self.change_state('PROBE_MINUS')
            return

        if self.state == 'PROBE_MINUS':
            probe = abs(
                float(self.get_parameter('probe_offset').value)
            )

            self.publish_pd(
                self.targets_from_delta(-probe)
            )
            self.collect_probe(self.minus_samples)

            if now - self.state_start >= float(
                self.get_parameter('probe_time').value
            ):
                self.finish_probe()
            return

        if self.state == 'LEVEL':
            gain = float(
                self.get_parameter('level_gain').value
            )
            damping = float(
                self.get_parameter('level_damping').value
            )
            max_delta = abs(
                float(self.get_parameter('max_delta').value)
            )

            delta_rate = self.control_sign * (
                gain * self.roll_f
                + damping * self.roll_rate
            )

            self.delta += delta_rate * dt
            self.delta = clamp(
                self.delta,
                -max_delta,
                max_delta
            )

            self.publish_pd(
                self.targets_from_delta(self.delta)
            )

            abs_roll = abs(self.roll_f)

            if abs_roll < self.best_abs_roll:
                self.best_abs_roll = abs_roll
                self.best_delta = self.delta

            roll_tol = math.radians(
                float(
                    self.get_parameter(
                        'roll_tolerance_deg'
                    ).value
                )
            )
            rate_tol = math.radians(
                float(
                    self.get_parameter(
                        'roll_rate_tolerance_deg_s'
                    ).value
                )
            )

            inside = (
                abs_roll <= roll_tol
                and abs(self.roll_rate) <= rate_tol
            )

            if inside:
                if self.good_since is None:
                    self.good_since = now
            else:
                self.good_since = None

            settle_time = float(
                self.get_parameter('settle_time').value
            )

            if (
                self.good_since is not None
                and now - self.good_since >= settle_time
            ):
                self.final_delta = self.delta
                self.save_result(converged=True)
                self.change_state('DONE')
                return

            timeout = float(
                self.get_parameter('timeout').value
            )

            if now - self.level_start >= timeout:
                self.final_delta = self.best_delta
                self.get_logger().warning(
                    'Timeout reached; saving best observed offset.'
                )
                self.save_result(converged=False)
                self.change_state('DONE')
            return

        if self.state == 'DONE':
            self.publish_pd(
                self.targets_from_delta(self.final_delta)
            )
            return

        if self.state == 'ERROR':
            self.publish_pd(self.nominal_targets())

    def stop(self):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LegRollCalibrator()

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
