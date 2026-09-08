"""Construct one registered environment and run finite zero-action steps."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils.app import run_with_simulation_app
from legged_lab.utils.task_registry import task_registry


def main():
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.scene.num_envs = 1
    cfg.device = args.device
    cfg.sim.device = args.device
    cfg.scene.seed = 42
    env = task_registry.get_task_class(args.task)(cfg, args.headless)
    observation, extras = env.get_observations()
    for _ in range(args.steps):
        observation, reward, reset, extras = env.step(
            torch.zeros((1, env.num_actions), device=env.device)
        )
        if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
            raise RuntimeError("non-finite smoke-test output")
    privileged = next(iter(extras["observations"].values()))
    print(
        "SMOKE_TEST_PASS "
        f"task={args.task} actions={env.num_actions} obs={observation.shape[1]} "
        f"privileged_obs={privileged.shape[1]} steps={args.steps}"
    )


if __name__ == "__main__":
    run_with_simulation_app(simulation_app, main)
