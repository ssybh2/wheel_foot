#!/usr/bin/env python3

from typing import Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from five_bar_kinematics import (
    FiveBarGeometry,
    FiveBarKinematics,
)


class FiveBarKinematicsNode(Node):
    ACTIVE_JOINTS = [
        'big_arm_1_joint',
        'big_arm_2_joint',
        'big_arm_3_joint',
        'big_arm_4_joint',
    ]

    def __init__(self) -> None:
        super().__init__('five_bar_kinematics')

        self.joint_positions: Dict[str, float] = {}

        # wheel 1:
        # big_arm_2_joint + big_arm_3_joint
        self.wheel_1_kinematics = FiveBarKinematics(
            FiveBarGeometry(
                base_a=(0.161440, 0.026313),
                base_b=(0.231306, 0.030643),

                upper_zero_a=(-0.070141, -0.038473),
                upper_zero_b=(0.077142, -0.021192),

                lower_length_a=0.120000,
                lower_length_b=0.120000,

                reference_wheel_center=(
                    0.204820,
                    -0.051057,
                ),
            )
        )

        # wheel 2:
        # big_arm_4_joint + big_arm_1_joint
        self.wheel_2_kinematics = FiveBarKinematics(
            FiveBarGeometry(
                base_a=(0.161440, 0.026313),
                base_b=(0.231306, 0.030643),

                upper_zero_a=(-0.071059, -0.036750),
                upper_zero_b=(0.074699, -0.028637),

                lower_length_a=0.120000,
                lower_length_b=0.120000,

                reference_wheel_center=(
                    0.201207,
                    -0.056453,
                ),
            )
        )

        self.fk_publisher = self.create_publisher(
            Float64MultiArray,
            '/kinematics/fk_wheel_centers',
            10,
        )

        self.ik_publisher = self.create_publisher(
            JointState,
            '/kinematics/ik_joint_targets',
            10,
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            20,
        )

        self.create_subscription(
            Float64MultiArray,
            '/kinematics/ik_request',
            self.ik_request_callback,
            10,
        )

        self.get_logger().info(
            'Five-bar kinematics node started'
        )

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        for name, position in zip(
            message.name,
            message.position,
        ):
            if name in self.ACTIVE_JOINTS:
                self.joint_positions[name] = float(position)

        if not all(
            name in self.joint_positions
            for name in self.ACTIVE_JOINTS
        ):
            return

        try:
            # wheel 1由joint 2和joint 3形成闭环
            wheel_1 = self.wheel_1_kinematics.forward(
                self.joint_positions['big_arm_2_joint'],
                self.joint_positions['big_arm_3_joint'],
            )

            # wheel 2由joint 4和joint 1形成闭环
            wheel_2 = self.wheel_2_kinematics.forward(
                self.joint_positions['big_arm_4_joint'],
                self.joint_positions['big_arm_1_joint'],
            )

        except ValueError as error:
            self.get_logger().warning(
                f'Forward kinematics failed: {error}'
            )
            return

        output = Float64MultiArray()

        # [wheel1_x, wheel1_z, wheel2_x, wheel2_z]
        output.data = [
            wheel_1[0],
            wheel_1[1],
            wheel_2[0],
            wheel_2[1],
        ]

        self.fk_publisher.publish(output)

    def ik_request_callback(
        self,
        message: Float64MultiArray,
    ) -> None:
        if len(message.data) != 4:
            self.get_logger().error(
                'IK request must contain: '
                '[wheel1_x, wheel1_z, '
                'wheel2_x, wheel2_z]'
            )
            return

        if not all(
            name in self.joint_positions
            for name in self.ACTIVE_JOINTS
        ):
            self.get_logger().warning(
                'Waiting for current joint states'
            )
            return

        wheel_1_target = (
            float(message.data[0]),
            float(message.data[1]),
        )

        wheel_2_target = (
            float(message.data[2]),
            float(message.data[3]),
        )

        try:
            q2, q3 = self.wheel_1_kinematics.inverse(
                wheel_1_target,
                (
                    self.joint_positions['big_arm_2_joint'],
                    self.joint_positions['big_arm_3_joint'],
                ),
            )

            q4, q1 = self.wheel_2_kinematics.inverse(
                wheel_2_target,
                (
                    self.joint_positions['big_arm_4_joint'],
                    self.joint_positions['big_arm_1_joint'],
                ),
            )

        except ValueError as error:
            self.get_logger().error(
                f'Inverse kinematics failed: {error}'
            )
            return

        output = JointState()
        output.header.stamp = self.get_clock().now().to_msg()

        output.name = [
            'big_arm_1_joint',
            'big_arm_2_joint',
            'big_arm_3_joint',
            'big_arm_4_joint',
        ]

        output.position = [
            q1,
            q2,
            q3,
            q4,
        ]

        self.ik_publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = FiveBarKinematicsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
