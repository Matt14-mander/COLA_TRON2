"""Nominal URDF forward kinematics for generating/verifying attachment frames."""

import math
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def link_transforms(urdf, joint_positions):
    model = ET.parse(urdf).getroot()
    joints = list(model.findall("joint"))
    children = {j.find("child").get("link") for j in joints}
    roots = {l.get("name") for l in model.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("Expected one URDF root")
    transforms = {roots.pop(): np.eye(4)}
    while joints:
        remaining = []
        for joint in joints:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                remaining.append(joint)
                continue
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
            transform = np.eye(4)
            transform[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
            transform[:3, 3] = xyz
            movement = np.eye(4)
            value = joint_positions.get(joint.get("name"), 0.0)
            axis = joint.find("axis")
            vector = [float(v) for v in axis.get("xyz").split()] if axis is not None else [1, 0, 0]
            if joint.get("type") in ("revolute", "continuous"):
                movement[:3, :3] = rotation(vector, value)
            elif joint.get("type") == "prismatic":
                movement[:3, 3] = np.asarray(vector) * value
            transforms[joint.find("child").get("link")] = transforms[parent] @ transform @ movement
        if len(remaining) == len(joints):
            raise ValueError("Disconnected or cyclic URDF")
        joints = remaining
    return transforms


def quaternion_wxyz(matrix):
    # Eigenvector method remains stable for rotations near pi.
    m = np.asarray(matrix)
    k = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[1, 0]+m[0, 1], m[2, 0]+m[0, 2], m[2, 1]-m[1, 2]],
        [m[1, 0]+m[0, 1], m[1, 1]-m[0, 0]-m[2, 2], m[2, 1]+m[1, 2], m[0, 2]-m[2, 0]],
        [m[2, 0]+m[0, 2], m[2, 1]+m[1, 2], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    quat = vectors[:, -1][[3, 0, 1, 2]]
    return (quat if quat[0] >= 0 else -quat).tolist()
