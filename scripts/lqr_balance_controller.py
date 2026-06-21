#!/usr/bin/env python3

import math
import time
from typing import Dict, Optional

import rclpy
from gazebo_msgs.msg import LinkStates
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import SetBool


WHEEL_JOINTS = ['wheel_1_joint', 'wheel_2_joint']


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quaternion_to_pitch(x: float, y: float, z: float, w: float) -> float:
    sin_pitch = 2.0 * (w * y - z * x)
    return math.asin(clamp(sin_pitch, -1.0, 1.0))


def slew(current: float, target: float, rate: float, dt: float) -> float:
    maximum_step = max(rate, 0.0) * max(dt, 0.0)
    error = target - current
    if abs(error) <= maximum_step:
        return target
    return current + math.copysign(maximum_step, error)


class LqrBalanceController(Node):
    """Four-state wheel LQR: [pitch, pitch_rate, position, velocity]."""

    def __init__(self) -> None:
        super().__init__('lqr_balance_controller')

        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('link_states_topic', '/gazebo/link_states')
        self.declare_parameter('fk_topic', '/kinematics/fk_wheel_centers')
        self.declare_parameter('leg_ready_topic', '/fixed_leg_hold/ready')
        self.declare_parameter(
            'wheel_command_topic',
            '/wheel_effort_controller/commands',
        )
        self.declare_parameter(
            'base_link_name',
            'wheel_leg_robot::base_link',
        )

        self.declare_parameter('control_rate', 200.0)
        self.declare_parameter('state_timeout', 0.20)
        self.declare_parameter('status_period', 0.20)
        self.declare_parameter('auto_enable', True)
        self.declare_parameter('enable_delay', 0.50)
        self.declare_parameter('require_leg_ready', False)

        self.declare_parameter('wheel_radius', 0.029)
        self.declare_parameter('pitch_zero_deg', 0.0)
        self.declare_parameter('pitch_sign', 1.0)
        self.declare_parameter('pitch_rate_sign', 1.0)
        self.declare_parameter('wheel_1_state_sign', 1.0)
        self.declare_parameter('wheel_2_state_sign', 1.0)
        self.declare_parameter('wheel_1_effort_sign', 1.0)
        self.declare_parameter('wheel_2_effort_sign', 1.0)
        self.declare_parameter('control_sign', 1.0)

        # Temporary fixed-height LQR gain from the assumed model:
        # m=3.0 kg, M=1.0 kg, h=0.14 m, I=0.04 kg m^2,
        # r=0.029 m, Ts=0.005 s, Q=diag(120,8,1,3), R=1.
        self.declare_parameter('k_theta', -10.43025064)
        self.declare_parameter('k_theta_dot', -1.81360324)
        self.declare_parameter('k_x', 0.62725848)
        self.declare_parameter('k_x_dot', 1.76654387)

        self.declare_parameter('max_total_wheel_torque', 0.30)
        self.declare_parameter('torque_slew_rate', 3.0)
        self.declare_parameter('fall_angle_deg', 30.0)

        self.declare_parameter('target_leg_height', 0.140)
        self.declare_parameter('max_leg_height_error', 0.040)
        self.declare_parameter('enforce_leg_height_safety', False)

        self.joint_state_topic = str(
            self.get_parameter('joint_state_topic').value
        )
        self.link_states_topic = str(
            self.get_parameter('link_states_topic').value
        )
        self.fk_topic = str(self.get_parameter('fk_topic').value)
        self.leg_ready_topic = str(
            self.get_parameter('leg_ready_topic').value
        )
        self.wheel_command_topic = str(
            self.get_parameter('wheel_command_topic').value
        )
        self.base_link_name = str(self.get_parameter('base_link_name').value)

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.state_timeout = float(self.get_parameter('state_timeout').value)
        self.status_period = float(self.get_parameter('status_period').value)
        self.auto_enable = bool(self.get_parameter('auto_enable').value)
        self.enable_delay = float(self.get_parameter('enable_delay').value)
        self.require_leg_ready = bool(
            self.get_parameter('require_leg_ready').value
        )

        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.pitch_zero = math.radians(
            float(self.get_parameter('pitch_zero_deg').value)
        )
        self.pitch_sign = float(self.get_parameter('pitch_sign').value)
        self.pitch_rate_sign = float(
            self.get_parameter('pitch_rate_sign').value
        )
        self.wheel_1_state_sign = float(
            self.get_parameter('wheel_1_state_sign').value
        )
        self.wheel_2_state_sign = float(
            self.get_parameter('wheel_2_state_sign').value
        )
        self.wheel_1_effort_sign = float(
            self.get_parameter('wheel_1_effort_sign').value
        )
        self.wheel_2_effort_sign = float(
            self.get_parameter('wheel_2_effort_sign').value
        )
        self.control_sign = float(self.get_parameter('control_sign').value)

        self.k_theta = float(self.get_parameter('k_theta').value)
        self.k_theta_dot = float(self.get_parameter('k_theta_dot').value)
        self.k_x = float(self.get_parameter('k_x').value)
        self.k_x_dot = float(self.get_parameter('k_x_dot').value)

        self.max_total_wheel_torque = float(
            self.get_parameter('max_total_wheel_torque').value
        )
        self.torque_slew_rate = float(
            self.get_parameter('torque_slew_rate').value
        )
        self.fall_angle = math.radians(
            float(self.get_parameter('fall_angle_deg').value)
        )

        self.target_leg_height = float(
            self.get_parameter('target_leg_height').value
        )
        self.max_leg_height_error = float(
            self.get_parameter('max_leg_height_error').value
        )
        self.enforce_leg_height_safety = bool(
            self.get_parameter('enforce_leg_height_safety').value
        )

        self.wheel_positions: Dict[str, float] = {}
        self.wheel_velocities: Dict[str, float] = {}
        self.last_joint_state_wall = 0.0

        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.last_link_state_wall = 0.0

        self.leg_ready = False
        self.current_leg_height: Optional[float] = None

        self.enabled = False
        self.position_reference: Optional[float] = None
        self.last_total_torque = 0.0

        self.start_wall = time.monotonic()
        self.last_loop_wall = self.start_wall
        self.last_status_wall = 0.0

        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            self.wheel_command_topic,
            20,
        )

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            50,
        )
        self.create_subscription(
            LinkStates,
            self.link_states_topic,
            self.link_states_callback,
            30,
        )
        self.create_subscription(
            Float64MultiArray,
            self.fk_topic,
            self.fk_callback,
            20,
        )
        self.create_subscription(
            Bool,
            self.leg_ready_topic,
            self.leg_ready_callback,
            20,
        )

        self.enable_service = self.create_service(
            SetBool,
            '/lqr_balance/enable',
            self.enable_callback,
        )

        self.timer = self.create_timer(
            1.0 / max(self.control_rate, 1.0),
            self.control_loop,
        )

        self.get_logger().info(
            'LQR balance controller started with K=['
            f'{self.k_theta:.6f}, {self.k_theta_dot:.6f}, '
            f'{self.k_x:.6f}, {self.k_x_dot:.6f}]'
        )

    def joint_state_callback(self, message: JointState) -> None:
        has_velocity = len(message.velocity) == len(message.name)
        for index, name in enumerate(message.name):
            if name not in WHEEL_JOINTS:
                continue
            if index < len(message.position):
                self.wheel_positions[name] = float(message.position[index])
            if has_velocity:
                self.wheel_velocities[name] = float(message.velocity[index])
        self.last_joint_state_wall = time.monotonic()

    def link_states_callback(self, message: LinkStates) -> None:
        try:
            index = message.name.index(self.base_link_name)
        except ValueError:
            return

        pose = message.pose[index]
        twist = message.twist[index]
        self.pitch = quaternion_to_pitch(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.pitch_rate = float(twist.angular.y)
        self.last_link_state_wall = time.monotonic()

    def fk_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) < 4:
            return
        wheel_1_z = float(message.data[1])
        wheel_2_z = float(message.data[3])
        self.current_leg_height = -0.5 * (wheel_1_z + wheel_2_z)

    def leg_ready_callback(self, message: Bool) -> None:
        self.leg_ready = bool(message.data)

    def state_is_ready(self, now: float) -> bool:
        if now - self.last_joint_state_wall > self.state_timeout:
            return False
        if now - self.last_link_state_wall > self.state_timeout:
            return False
        return all(name in self.wheel_positions for name in WHEEL_JOINTS)

    def current_position_and_velocity(self) -> tuple[float, float]:
        q1 = self.wheel_1_state_sign * self.wheel_positions['wheel_1_joint']
        q2 = self.wheel_2_state_sign * self.wheel_positions['wheel_2_joint']
        dq1 = self.wheel_1_state_sign * self.wheel_velocities.get(
            'wheel_1_joint',
            0.0,
        )
        dq2 = self.wheel_2_state_sign * self.wheel_velocities.get(
            'wheel_2_joint',
            0.0,
        )
        position = 0.5 * self.wheel_radius * (q1 + q2)
        velocity = 0.5 * self.wheel_radius * (dq1 + dq2)
        return position, velocity

    def enable_callback(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        now = time.monotonic()
        if request.data:
            if not self.state_is_ready(now):
                response.success = False
                response.message = 'Cannot enable: wheel/link state is not ready.'
                return response
            position, _ = self.current_position_and_velocity()
            self.position_reference = position
            self.enabled = True
            self.last_total_torque = 0.0
            response.success = True
            response.message = 'LQR enabled and position reference captured.'
        else:
            self.enabled = False
            self.position_reference = None
            self.last_total_torque = 0.0
            self.publish_zero()
            response.success = True
            response.message = 'LQR disabled.'
        return response

    def publish_zero(self) -> None:
        message = Float64MultiArray()
        message.data = [0.0, 0.0]
        self.command_publisher.publish(message)

    def control_loop(self) -> None:
        now = time.monotonic()
        dt = clamp(now - self.last_loop_wall, 1.0e-4, 0.05)
        self.last_loop_wall = now

        if not self.state_is_ready(now):
            self.last_total_torque = 0.0
            self.publish_zero()
            return

        if (
            self.auto_enable
            and not self.enabled
            and now - self.start_wall >= self.enable_delay
        ):
            position, _ = self.current_position_and_velocity()
            self.position_reference = position
            self.enabled = True
            self.last_total_torque = 0.0
            self.get_logger().warning(
                'Auto-enabled LQR. If correction direction is wrong, '
                'set control_sign to -1.0.'
            )

        if not self.enabled:
            self.last_total_torque = 0.0
            self.publish_zero()
            return

        if self.require_leg_ready and not self.leg_ready:
            self.last_total_torque = 0.0
            self.publish_zero()
            return

        if (
            self.enforce_leg_height_safety
            and self.current_leg_height is not None
            and abs(self.current_leg_height - self.target_leg_height)
            > self.max_leg_height_error
        ):
            self.last_total_torque = 0.0
            self.publish_zero()
            return

        theta = self.pitch_sign * (self.pitch - self.pitch_zero)
        theta_dot = self.pitch_rate_sign * self.pitch_rate
        position, velocity = self.current_position_and_velocity()

        if self.position_reference is None:
            self.position_reference = position

        position_error = position - self.position_reference

        if abs(theta) >= self.fall_angle:
            self.enabled = False
            self.last_total_torque = 0.0
            self.publish_zero()
            self.get_logger().error(
                f'Fall protection: pitch={math.degrees(theta):.1f} deg. '
                'LQR disabled.'
            )
            return

        state_feedback = (
            self.k_theta * theta
            + self.k_theta_dot * theta_dot
            + self.k_x * position_error
            + self.k_x_dot * velocity
        )
        target_total_torque = self.control_sign * (-state_feedback)
        target_total_torque = clamp(
            target_total_torque,
            -self.max_total_wheel_torque,
            self.max_total_wheel_torque,
        )

        total_torque = slew(
            self.last_total_torque,
            target_total_torque,
            self.torque_slew_rate,
            dt,
        )
        self.last_total_torque = total_torque

        wheel_1_torque = (
            0.5 * self.wheel_1_effort_sign * total_torque
        )
        wheel_2_torque = (
            0.5 * self.wheel_2_effort_sign * total_torque
        )

        message = Float64MultiArray()
        message.data = [wheel_1_torque, wheel_2_torque]
        self.command_publisher.publish(message)

        if now - self.last_status_wall >= self.status_period:
            leg_height_text = (
                'unknown'
                if self.current_leg_height is None
                else f'{self.current_leg_height:.3f}'
            )
            self.get_logger().info(
                f'enabled={self.enabled} leg_ready={self.leg_ready} '
                f'theta={math.degrees(theta):+.2f} deg '
                f'theta_dot={theta_dot:+.3f} rad/s '
                f'x_err={position_error:+.4f} m '
                f'v={velocity:+.4f} m/s '
                f'u_total={total_torque:+.3f} N*m '
                f'h={leg_height_text} m'
            )
            self.last_status_wall = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LqrBalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_zero()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
