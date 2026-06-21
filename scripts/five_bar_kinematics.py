#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


PointXZ = Tuple[float, float]


def wrap_to_pi(angle: float) -> float:
    """将角度限制到[-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angular_distance(angle_a: float, angle_b: float) -> float:
    return abs(wrap_to_pi(angle_a - angle_b))


@dataclass(frozen=True)
class FiveBarGeometry:
    # 两个主动关节在base_link坐标系X-Z平面中的位置
    base_a: PointXZ
    base_b: PointXZ

    # q=0时，主动关节到膝关节的X-Z向量
    upper_zero_a: PointXZ
    upper_zero_b: PointXZ

    # 两个下连杆长度
    lower_length_a: float
    lower_length_b: float

    # 用于选择正确闭环装配支路
    reference_wheel_center: PointXZ


class FiveBarKinematics:
    def __init__(self, geometry: FiveBarGeometry) -> None:
        self.geometry = geometry
        self.last_fk_point: Optional[PointXZ] = (
            geometry.reference_wheel_center
        )

    @staticmethod
    def _rotate_about_positive_y(
        vector: PointXZ,
        joint_angle: float,
    ) -> PointXZ:
        """
        Gazebo关节轴为+Y。

        R_y(q)作用于X-Z平面：
            x' = cos(q)x + sin(q)z
            z' = -sin(q)x + cos(q)z
        """
        x_value, z_value = vector
        cosine = math.cos(joint_angle)
        sine = math.sin(joint_angle)

        return (
            cosine * x_value + sine * z_value,
            -sine * x_value + cosine * z_value,
        )

    @staticmethod
    def _circle_intersections(
        center_a: PointXZ,
        radius_a: float,
        center_b: PointXZ,
        radius_b: float,
    ) -> List[PointXZ]:
        x_a, z_a = center_a
        x_b, z_b = center_b

        dx = x_b - x_a
        dz = z_b - z_a
        distance = math.hypot(dx, dz)

        if distance < 1.0e-9:
            raise ValueError("两个膝关节重合，正运动学不可解")

        if distance > radius_a + radius_b + 1.0e-9:
            raise ValueError("下连杆圆不相交")

        if distance < abs(radius_a - radius_b) - 1.0e-9:
            raise ValueError("一个下连杆圆完全包含另一个圆")

        along = (
            radius_a * radius_a
            - radius_b * radius_b
            + distance * distance
        ) / (2.0 * distance)

        height_squared = radius_a * radius_a - along * along

        if height_squared < -1.0e-9:
            raise ValueError("闭环机构几何计算出现负根号")

        height = math.sqrt(max(0.0, height_squared))

        middle_x = x_a + along * dx / distance
        middle_z = z_a + along * dz / distance

        perpendicular_x = -dz / distance
        perpendicular_z = dx / distance

        return [
            (
                middle_x + height * perpendicular_x,
                middle_z + height * perpendicular_z,
            ),
            (
                middle_x - height * perpendicular_x,
                middle_z - height * perpendicular_z,
            ),
        ]

    def forward(
        self,
        joint_angle_a: float,
        joint_angle_b: float,
    ) -> PointXZ:
        """
        正运动学：
            两个主动关节角 -> 轮轴中心X、Z坐标
        """
        geometry = self.geometry

        upper_a = self._rotate_about_positive_y(
            geometry.upper_zero_a,
            joint_angle_a,
        )
        upper_b = self._rotate_about_positive_y(
            geometry.upper_zero_b,
            joint_angle_b,
        )

        knee_a = (
            geometry.base_a[0] + upper_a[0],
            geometry.base_a[1] + upper_a[1],
        )
        knee_b = (
            geometry.base_b[0] + upper_b[0],
            geometry.base_b[1] + upper_b[1],
        )

        candidates = self._circle_intersections(
            knee_a,
            geometry.lower_length_a,
            knee_b,
            geometry.lower_length_b,
        )

        reference = (
            self.last_fk_point
            if self.last_fk_point is not None
            else geometry.reference_wheel_center
        )

        selected = min(
            candidates,
            key=lambda point: math.hypot(
                point[0] - reference[0],
                point[1] - reference[1],
            ),
        )

        self.last_fk_point = selected
        return selected

    @staticmethod
    def _single_chain_inverse(
        base: PointXZ,
        upper_zero: PointXZ,
        lower_length: float,
        target: PointXZ,
        seed_joint_angle: float,
    ) -> float:
        upper_length = math.hypot(
            upper_zero[0],
            upper_zero[1],
        )

        dx = target[0] - base[0]
        dz = target[1] - base[1]
        target_distance = math.hypot(dx, dz)

        minimum_reach = abs(upper_length - lower_length)
        maximum_reach = upper_length + lower_length

        if (
            target_distance < minimum_reach - 1.0e-9
            or target_distance > maximum_reach + 1.0e-9
        ):
            raise ValueError(
                f"目标点不可达，距离={target_distance:.6f} m，"
                f"允许范围=[{minimum_reach:.6f}, "
                f"{maximum_reach:.6f}] m"
            )

        cosine_delta = (
            upper_length * upper_length
            + target_distance * target_distance
            - lower_length * lower_length
        ) / (2.0 * upper_length * target_distance)

        cosine_delta = max(-1.0, min(1.0, cosine_delta))

        target_direction = math.atan2(dz, dx)
        delta = math.acos(cosine_delta)

        zero_direction = math.atan2(
            upper_zero[1],
            upper_zero[0],
        )

        # 两个二连杆逆解分支
        upper_directions = [
            target_direction + delta,
            target_direction - delta,
        ]

        joint_candidates = [
            wrap_to_pi(zero_direction - direction)
            for direction in upper_directions
        ]

        # 选择最接近当前关节角的分支，避免突然翻转
        return min(
            joint_candidates,
            key=lambda angle: angular_distance(
                angle,
                seed_joint_angle,
            ),
        )

    def inverse(
        self,
        target: PointXZ,
        seed_joint_angles: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        逆运动学：
            轮轴目标X、Z -> 两个主动关节目标角
        """
        geometry = self.geometry

        joint_a = self._single_chain_inverse(
            geometry.base_a,
            geometry.upper_zero_a,
            geometry.lower_length_a,
            target,
            seed_joint_angles[0],
        )

        joint_b = self._single_chain_inverse(
            geometry.base_b,
            geometry.upper_zero_b,
            geometry.lower_length_b,
            target,
            seed_joint_angles[1],
        )

        return joint_a, joint_b
