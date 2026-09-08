# TRON2 migration status

The selected target is **SFYG_TRON2A**: ten leg joints, one six-axis arm and
two gripper joints. The policy controls the ten leg joints and six arm joints;
the gripper remains at its configured grasp width.

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
- The repository now includes the Apache-2.0 SFYG URDF, USD, MJCF and meshes,
  with the source NOTICE and third-party notices under `assets/tron2`.
- Five lazy task registrations cover Phase 1 and fixed-bar/no-object variants
  for Phases 2 and 3. The fixed-bar population uses one fixed constraint from
  `arm6_Link` to the robot endpoint; the opposite endpoint is human-controlled.
- The custom residual and distillation networks derive the base observation
  from the configured privileged tail instead of assuming 29 G1 joints.
- `deployment/mujoco/run_sfyg_sim2sim.py` supports Phase-1 and mass-conditioned
  Phase-3 TorchScript policies, including Windows paths containing non-ASCII
  characters.

## Model inventory

The supplied description repository contains DA (14 joints), DACH (16), SF
(10), WF (10), SFYG (18) and WFYG (18). These are movable URDF joint counts,
not automatically policy action counts. DA/DACH contain arms without legs;
SF/WF contain legs without arms; YG models contain one six-axis arm and two
gripper joints. Do not infer a G1-like two-arm/two-leg robot from the name.
See `tron2_asset_inventory.json` for source paths and exact names.

## Validated contracts

| Phase | Frame/history | Network input | Output |
|---|---:|---:|---:|
| Phase 1 actor | 65 x 10 | 650 | 16 |
| Phase 1 critic | 70 x 10 | 700 | 1 |
| Phase 2 actor | 85 x 10 | 850 | 16 |
| Phase 2 critic | 90 x 10 | 900 | 1 |
| Phase 3 student | 66 x 25 | 1650 | 16 |
| Phase 3 teacher | 85 x 10 | 850 | 16 |

Eleven CPU tests validate joint ordering/action scatter, asset inspection, the
three checkpoint transitions, privileged-tail removal and 16-action outputs.
An Isaac smoke run completed one Phase-1 PPO iteration (24 transitions) on the
real SFYG USD. The fixed-bar scene constructed successfully and reached its
first controller step. Full fixed-bar stepping is still to be repeated under
the validated Isaac Sim 5.1 environment: this development host has 4.5, where
the first Warp CUDA compilation stalled after scene creation.

The bundled MJCF was also checked against all 18 movable profile joints. Both
the `650 -> 16` Phase-1 path and `1650 -> 16` Phase-3 path completed finite
headless rollouts. A two-second zero-action Phase-1 smoke policy fell to a base
height of 0.1312 m; this is expected from an untrained open-loop biped and
confirms that learned balance is still required.

## Training commands

Phase 1:

```bash
torchrun --standalone --nproc_per_node=8 \
  legged_lab/scripts/train_locomotion.py \
  --task=tron2_sfyg_phase_1_locomotion \
  --distributed --headless --num_envs=2048 --max_iterations=10000 \
  --logger=wandb --run_name=sfyg_phase1
```

Phase 2 uses six single-arm fixed-bar ranks and two no-object ranks:

```bash
torchrun --standalone --nproc_per_node=8 \
  legged_lab/scripts/train_collaboration.py --robot=tron2_sfyg --phase=2 \
  --distributed --headless --num_envs=2048 --max_iterations=8000 \
  --no_object_rank_count=2 --right_fixed_rank_count=0 \
  --resume=True --load_run="$PHASE1_RUN" --checkpoint="$PHASE1_CHECKPOINT" \
  --logger=wandb --run_name=sfyg_phase2
```

Phase 3 uses the same population split:

```bash
torchrun --standalone --nproc_per_node=8 \
  legged_lab/scripts/train_collaboration.py --robot=tron2_sfyg --phase=3 \
  --distributed --headless --num_envs=2048 --max_iterations=16000 \
  --no_object_rank_count=2 --right_fixed_rank_count=0 \
  --resume=True --load_run="$PHASE2_RUN" --checkpoint="$PHASE2_CHECKPOINT" \
  --logger=wandb --run_name=sfyg_phase3
```

For a two-GPU functional run, use `--nproc_per_node=2
--no_object_rank_count=1 --right_fixed_rank_count=0`. Run a one-environment
check before training:

```bash
python legged_lab/scripts/smoke_test.py \
  --task=tron2_sfyg_phase_2_teacher_fixed_bar --headless --steps=8
```

Export and run a student in MuJoCo:

```bash
python deployment/mujoco/export_policy.py \
  --checkpoint /path/model.pt --output /path/sfyg_student.jit --num-actions=16
python deployment/mujoco/run_sfyg_sim2sim.py \
  --policy /path/sfyg_student.jit --headless --duration=20
```

On Windows, set `WARP_CACHE_PATH` to a writable path containing ASCII
characters before launching Isaac Sim. Dot-sourcing `setup_env.ps1` configures
this automatically when `%LOCALAPPDATA%` is ASCII-only.

## Remaining physical validation

1. Verify `sfyg_profile.json` gains, nominal pose, gripper width, action scale
   and torque limits against the exact physical robot. They are simulation
   defaults derived from the provided models and local controller examples,
   not factory calibration values.
2. Inspect the first Isaac reset visually and measure the `arm6_Link` grasp
   point. Adjust the attachment transform if the physical fixture differs from
   the nominal `gripper_pick` offset.
3. Tune the single-arm bar rewards and controller gains with short Phase-2
   runs before allocating a full distributed training job.
4. Validate exported policies in MuJoCo, then connect the vendor SDK with the
   exact 16-joint order from `sfyg_profile.json` before hardware testing.
