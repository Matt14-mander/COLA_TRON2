# TRON2 migration status

This is an in-progress port, not a runnable TRON2 training release.

## Implemented

- Locomotion, fixed-bar and no-object observations use explicit
  `cfg.robot.policy_joint_names` ordering instead of slicing the first 29 joints.
- Action targets are scattered into the complete articulation; uncontrolled
  joints retain their default positions. G1 retains its legacy ordering.
- Observation noise offsets follow the selected action count.
- `legged_lab.assets.tron2.make_tron2_asset` builds a USD articulation with
  explicitly supplied actuator parameters. It does not invent gains or select
  a hardware variant. Wheel variants require a different control adapter.
- `python -m legged_lab.scripts.inspect_tron2 --description-root PATH`
  inspects URDF joints/limits and model files without Isaac or torch.
  `--output PATH` writes JSON. ROS package aliases are recorded explicitly;
  successful mesh lookup does not mean the URDF importer is configured.

## Model inventory

The supplied description repository contains DA (14 joints), DACH (16), SF
(10), WF (10), SFYG (18) and WFYG (18). These are movable URDF joint counts,
not automatically policy action counts. DA/DACH contain arms without legs;
SF/WF contain legs without arms; YG models contain one six-axis arm and two
gripper joints. Do not infer a G1-like two-arm/two-leg robot from the name.
See `tron2_asset_inventory.json` for source paths and exact names.

## Pending hardware decision and work

1. Select actual variant (or provide the assembled model), default pose, gains,
   control modes, policy joints and task end-effector frames.
2. Adapt rewards, command generators, feet/contact selections and object
   constraints to that topology. A single-arm model cannot use the released
   two-wrist constraints unchanged.
3. Populate IsaacLab and the custom COLA_rsl_rl submodules; inspect and remove
   network/runner assumptions about G1 action and observation dimensions.
4. Register TRON2 tasks and add robot selection to training/evaluation once
   the actual environment configuration exists.
5. Adapt policy export and MuJoCo runner, then validate reset/step, dimensions,
   actuator order, training/checkpoint transitions and sim2sim.

No TRON2 task is registered yet. Existing commands and bundled student.jit
still target G1. Isaac simulation and tensor execution were not available in
the editing environment; syntax and simulator-independent tests are checked.
