#!/usr/bin/env python3
"""Run a Phase-1 or Phase-3 SFYG policy in the bundled robot-only MJCF."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time
import os

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "assets/tron2/tron2/SFYG_TRON2A/xml/robot.xml"
DEFAULT_PROFILE = ROOT / "assets/tron2/sfyg_profile.json"
SIM_DT = .001
POLICY_DT = .02
DECIMATION = round(POLICY_DT / SIM_DT)
ANGULAR_VELOCITY_SCALE = .25
JOINT_VELOCITY_SCALE = .05


def native_path(path):
    """Use an ASCII 8.3 path for native Windows loaders."""
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.isascii():
        return resolved
    import ctypes
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(resolved, buffer, len(buffer)):
        raise OSError(f"Could not create a Windows short path for {resolved}")
    return buffer.value


def rotate_inverse(q, vector):
    w, x, y, z = q
    rotation = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])
    return rotation.T @ vector


class Simulation:
    def __init__(self, args):
        self.profile = json.loads(args.profile.read_text(encoding="utf-8"))
        self.policy_names = self.profile["policy_joint_names"]
        self.num_actions = len(self.policy_names)
        self.model = mujoco.MjModel.from_xml_path(native_path(args.model))
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.device = torch.device(args.device)
        self.policy = torch.jit.load(native_path(args.policy), map_location=self.device).eval()

        movable = []
        for joint_id in range(self.model.njnt):
            kind = self.model.jnt_type[joint_id]
            if kind in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                movable.append(joint_id)
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in movable]
        if set(names) != set(self.profile["joint_parameters"]):
            raise RuntimeError("MJCF movable joints do not match the SFYG profile")
        name_to_id = dict(zip(names, movable))
        self.policy_joint_ids = [name_to_id[name] for name in self.policy_names]
        self.qpos = np.array([self.model.jnt_qposadr[i] for i in self.policy_joint_ids])
        self.dof = np.array([self.model.jnt_dofadr[i] for i in self.policy_joint_ids])
        actuator_by_joint = {int(self.model.actuator_trnid[i, 0]): i for i in range(self.model.nu)}
        self.actuators = np.array([actuator_by_joint[i] for i in self.policy_joint_ids])
        free = [i for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise RuntimeError("SFYG MJCF must have exactly one floating-base joint")
        self.base_qpos = int(self.model.jnt_qposadr[free[0]])
        self.base_dof = int(self.model.jnt_dofadr[free[0]])

        parameters = self.profile["joint_parameters"]
        self.defaults = np.array([parameters[n]["default_position"] for n in self.policy_names])
        self.kp = np.array([parameters[n]["stiffness"] for n in self.policy_names])
        self.kd = np.array([parameters[n]["damping"] for n in self.policy_names])
        self.effort = np.array([parameters[n]["effort_limit"] for n in self.policy_names])
        self.action_scale = float(self.profile["action_scale"])
        self.pose = np.asarray(self.profile["ee_quaternion_base"] + self.profile["ee_position_base"], dtype=np.float32)
        default_command = (0., 0., 0., self.profile["base_position"][2])
        self.command = np.asarray(args.command or default_command, dtype=np.float32)
        self.mass = float(args.mass_observation_kg)
        self.kind, self.history_length, self.frame_width = self.detect_contract()
        if self.kind == "student":
            if args.command is not None:
                raise ValueError("Phase-3 students use their masked training command; omit --command")
            self.command = np.array([0., 0., 0., self.profile["base_position"][2]], dtype=np.float32)
        else:
            limits = ((-.3, .5), (-.2, .2), (-.5, .5))
            height = self.profile["base_position"][2]
            limits += ((height - .04, height + .04),)
            for value, (lower, upper), label in zip(
                self.command, limits, ("forward velocity", "lateral velocity", "yaw velocity", "height")
            ):
                if not lower <= value <= upper:
                    raise ValueError(f"{label} {value:g} is outside the training range [{lower:g}, {upper:g}]")
        mass_lower, mass_upper = self.profile["bar_mass_range"]
        if not mass_lower - .5 <= self.mass <= mass_upper + .5:
            raise ValueError(
                f"mass observation {self.mass:g} is outside the training support "
                f"[{mass_lower - .5:g}, {mass_upper + .5:g}] kg"
            )
        self.history = deque(maxlen=self.history_length)
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target = self.defaults.copy()
        self.reset()

    @torch.inference_mode()
    def detect_contract(self):
        candidates = (("phase1", 10, 65), ("student", 25, 66))
        matches = []
        for item in candidates:
            try:
                result = self.policy(torch.zeros(1, item[1]*item[2], device=self.device))
                if tuple(result.shape) == (1, self.num_actions):
                    matches.append(item)
            except (RuntimeError, ValueError):
                pass
        if len(matches) != 1:
            raise RuntimeError(f"Expected one SFYG policy input contract; matched {matches}")
        return matches[0]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_qpos:self.base_qpos+7] = (*self.profile["base_position"], 1., 0., 0., 0.)
        for name, values in self.profile["joint_parameters"].items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = values["default_position"]
        self.data.qvel.fill(0.)
        self.data.ctrl.fill(0.)
        mujoco.mj_forward(self.model, self.data)
        self.previous_action.fill(0.)
        self.target[:] = self.defaults
        self.history.clear()

    def frame(self):
        q = self.data.qpos[self.qpos] - self.defaults
        dq = self.data.qvel[self.dof]
        quat = self.data.qpos[self.base_qpos+3:self.base_qpos+7]
        omega = self.data.qvel[self.base_dof+3:self.base_dof+6]
        pieces = [self.command, self.pose, omega*ANGULAR_VELOCITY_SCALE,
                  rotate_inverse(quat, np.array([0., 0., -1.])), q,
                  dq*JOINT_VELOCITY_SCALE, self.previous_action]
        if self.kind == "student":
            pieces.append(np.array([self.mass]))
        frame = np.concatenate(pieces).astype(np.float32)
        if frame.shape != (self.frame_width,):
            raise RuntimeError(f"Unexpected observation frame: {frame.shape}")
        return frame

    @torch.inference_mode()
    def policy_step(self):
        frame = self.frame()
        if not self.history:
            self.history.extend(frame.copy() for _ in range(self.history_length))
        else:
            self.history.append(frame)
        obs = torch.from_numpy(np.clip(np.concatenate(self.history), -100., 100.)).unsqueeze(0).to(self.device)
        action = self.policy(obs).squeeze(0).cpu().numpy()
        if action.shape != (self.num_actions,) or not np.isfinite(action).all():
            raise RuntimeError("Policy returned an invalid action")
        action = np.clip(action, -5., 5.)
        self.previous_action[:] = action
        self.target[:] = self.defaults + self.action_scale * action

    def physics_step(self):
        torque = self.kp*(self.target-self.data.qpos[self.qpos]) - self.kd*self.data.qvel[self.dof]
        self.data.ctrl[self.actuators] = np.clip(torque, -self.effort, self.effort)
        mujoco.mj_step(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--command", nargs=4, type=float)
    parser.add_argument("--mass-observation-kg", type=float, default=1.4)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    sim = Simulation(args)
    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    steps = round(args.duration / SIM_DT)
    start = time.perf_counter()
    for step in range(steps):
        if step % DECIMATION == 0:
            sim.policy_step()
        sim.physics_step()
        if viewer is not None:
            viewer.sync()
            target = start + (step + 1) * SIM_DT
            time.sleep(max(0., target - time.perf_counter()))
    if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
        raise RuntimeError("MuJoCo state became non-finite")
    base_height = float(sim.data.qpos[sim.base_qpos + 2])
    if viewer is not None:
        viewer.close()
    print(
        f"SFYG_SIM2SIM_PASS kind={sim.kind} steps={steps} "
        f"actions={sim.num_actions} base_height={base_height:.4f}"
    )


if __name__ == "__main__":
    main()
