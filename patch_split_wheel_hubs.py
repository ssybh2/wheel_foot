#!/usr/bin/env python3
"""Patch the Gazebo SDF so each wheel has a non-rotating hub carrier.

Before:
    leg link -- wheel_i_joint --> wheel_motor_i_link
    other leg -- close_loop_leg_i_joint --> wheel_motor_i_link

The rotating wheel link is also the four-bar loop closure body.

After:
    leg link -- hub_leg_i_joint --> wheel_hub_i_link
    other leg -- close_loop_leg_i_joint --> wheel_hub_i_link
    wheel_hub_i_link -- wheel_i_joint --> wheel_motor_i_link

The ros2_control joint names wheel_1_joint / wheel_2_joint are preserved.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional


def find_named(parent: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    for element in parent.findall(tag):
        if element.get("name") == name:
            return element
    return None


def child_text(parent: ET.Element, tag: str) -> str:
    element = parent.find(tag)
    if element is None or element.text is None:
        raise RuntimeError(
            f"Element <{parent.tag} name={parent.get('name')!r}> "
            f"has no <{tag}> text"
        )
    return element.text.strip()


def set_child_text(parent: ET.Element, tag: str, value: str) -> None:
    element = parent.find(tag)
    if element is None:
        element = ET.SubElement(parent, tag)
    element.text = value


def make_hub_link(hub_name: str, hub_joint_name: str) -> ET.Element:
    link = ET.Element("link", {"name": hub_name})

    pose = ET.SubElement(
        link,
        "pose",
        {"relative_to": hub_joint_name},
    )
    pose.text = "0 0 0 0 0 0"

    inertial = ET.SubElement(link, "inertial")
    inertial_pose = ET.SubElement(inertial, "pose")
    inertial_pose.text = "0 0 0 0 0 0"

    mass = ET.SubElement(inertial, "mass")
    mass.text = "0.05"

    inertia = ET.SubElement(inertial, "inertia")
    for name, value in (
        ("ixx", "0.00005"),
        ("ixy", "0"),
        ("ixz", "0"),
        ("iyy", "0.00005"),
        ("iyz", "0"),
        ("izz", "0.00005"),
    ):
        item = ET.SubElement(inertia, name)
        item.text = value

    return link


def set_passive_hub_joint_axis(joint: ET.Element) -> None:
    axis = joint.find("axis")
    if axis is None:
        axis = ET.SubElement(joint, "axis")

    xyz = axis.find("xyz")
    if xyz is None:
        xyz = ET.SubElement(axis, "xyz")
    xyz.text = "0 1 0"

    limit = axis.find("limit")
    if limit is None:
        limit = ET.SubElement(axis, "limit")

    set_child_text(limit, "lower", "-6.28318")
    set_child_text(limit, "upper", "6.28318")

    dynamics = axis.find("dynamics")
    if dynamics is None:
        dynamics = ET.SubElement(axis, "dynamics")

    set_child_text(dynamics, "damping", "0.03")
    set_child_text(dynamics, "friction", "0.005")

    for tag in ("spring_reference", "spring_stiffness"):
        element = dynamics.find(tag)
        if element is not None:
            dynamics.remove(element)


def make_wheel_joint(
    wheel_joint_name: str,
    hub_name: str,
    wheel_link_name: str,
    original_axis: ET.Element,
) -> ET.Element:
    joint = ET.Element(
        "joint",
        {
            "name": wheel_joint_name,
            "type": "revolute",
        },
    )

    pose = ET.SubElement(
        joint,
        "pose",
        {"relative_to": hub_name},
    )
    pose.text = "0 0 0 0 0 0"

    parent = ET.SubElement(joint, "parent")
    parent.text = hub_name

    child = ET.SubElement(joint, "child")
    child.text = wheel_link_name

    joint.append(copy.deepcopy(original_axis))
    return joint


def patch_one_side(model: ET.Element, index: int) -> None:
    wheel_joint_name = f"wheel_{index}_joint"
    wheel_link_name = f"wheel_motor_{index}_link"
    hub_name = f"wheel_hub_{index}_link"
    hub_joint_name = f"hub_leg_{index}_joint"
    close_loop_name = f"close_loop_leg_{index}_joint"

    if find_named(model, "link", hub_name) is not None:
        raise RuntimeError(
            f"{hub_name} already exists. The SDF appears already patched."
        )

    old_wheel_joint = find_named(
        model,
        "joint",
        wheel_joint_name,
    )
    if old_wheel_joint is None:
        raise RuntimeError(f"Cannot find joint {wheel_joint_name}")

    wheel_link = find_named(model, "link", wheel_link_name)
    if wheel_link is None:
        raise RuntimeError(f"Cannot find link {wheel_link_name}")

    close_loop_joint = find_named(
        model,
        "joint",
        close_loop_name,
    )
    if close_loop_joint is None:
        raise RuntimeError(f"Cannot find joint {close_loop_name}")

    old_parent = child_text(old_wheel_joint, "parent")
    old_child = child_text(old_wheel_joint, "child")
    if old_child != wheel_link_name:
        raise RuntimeError(
            f"{wheel_joint_name} child is {old_child}, expected "
            f"{wheel_link_name}"
        )

    original_axis = old_wheel_joint.find("axis")
    if original_axis is None:
        raise RuntimeError(f"{wheel_joint_name} has no <axis>")

    # 1. Convert the original wheel joint into a passive leg-to-hub joint.
    old_wheel_joint.set("name", hub_joint_name)
    old_wheel_joint.set("type", "revolute")
    set_child_text(old_wheel_joint, "parent", old_parent)
    set_child_text(old_wheel_joint, "child", hub_name)
    set_passive_hub_joint_axis(old_wheel_joint)

    # 2. Add the non-rotating hub carrier at the same wheel-center pose.
    hub_link = make_hub_link(hub_name, hub_joint_name)
    model.append(hub_link)

    # 3. Close the four-bar loop on the hub carrier, not the wheel.
    set_child_text(close_loop_joint, "child", hub_name)

    # 4. Add a new actuator joint from hub carrier to rotating wheel.
    new_wheel_joint = make_wheel_joint(
        wheel_joint_name,
        hub_name,
        wheel_link_name,
        original_axis,
    )
    model.append(new_wheel_joint)

    # The wheel link was originally posed relative to wheel_i_joint.
    # This reference is intentionally kept; it now points to the new
    # actuator joint.
    wheel_pose = wheel_link.find("pose")
    if wheel_pose is None:
        wheel_pose = ET.Element(
            "pose",
            {"relative_to": wheel_joint_name},
        )
        wheel_pose.text = "0 0 0 0 0 0"
        wheel_link.insert(0, wheel_pose)
    else:
        wheel_pose.set("relative_to", wheel_joint_name)
        wheel_pose.text = "0 0 0 0 0 0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sdf",
        nargs="?",
        default=(
            "~/foot_ws/src/wheel_leg_description/"
            "sdf/wheel_leg_robot_closed_loop_control.sdf"
        ),
        help="Path to wheel_leg_robot_closed_loop_control.sdf",
    )
    args = parser.parse_args()

    sdf_path = Path(args.sdf).expanduser().resolve()
    if not sdf_path.is_file():
        print(f"ERROR: SDF file not found: {sdf_path}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = sdf_path.with_suffix(
        sdf_path.suffix + f".backup_{timestamp}"
    )
    shutil.copy2(sdf_path, backup_path)

    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()

        model = root.find("model")
        if model is None:
            raise RuntimeError("Cannot find top-level <model>")

        patch_one_side(model, 1)
        patch_one_side(model, 2)

        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass

        tree.write(
            sdf_path,
            encoding="utf-8",
            xml_declaration=False,
        )

    except Exception:
        shutil.copy2(backup_path, sdf_path)
        print(
            f"Patch failed. Original restored from:\n  {backup_path}",
            file=sys.stderr,
        )
        raise

    print("Patch completed.")
    print(f"Modified:\n  {sdf_path}")
    print(f"Backup:\n  {backup_path}")
    print()
    print("Expected new topology:")
    print("  leg -> hub_leg_i_joint -> wheel_hub_i_link")
    print("  other leg -> close_loop_leg_i_joint -> wheel_hub_i_link")
    print("  wheel_hub_i_link -> wheel_i_joint -> wheel_motor_i_link")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
