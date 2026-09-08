"""SFYG 10-leg + 6-arm action configurations for the three COLA phases."""

from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.assets.tron2.sfyg import load_profile, make_sfyg_asset
from legged_lab.envs.base import collaboration_mdp
from legged_lab.envs.base.common_config import (
    LocomotionExperimentCfg, LocomotionPoseTrajectoryCfg, TerminationObservationCfg,
)
from legged_lab.envs.base.locomotion_config import LocomotionEnvCfg
from legged_lab.envs.g1.phase_1_locomotion_config import Phase1LocomotionAgentCfg
from legged_lab.envs.base.collaboration_config import (
    CollaborationEnvCfg, TeacherAgentCfg, StudentAgentCfg, teacher_robot_cfg, student_robot_cfg,
    collaboration_experiment_cfg, support_height_control_cfg, bar_controller_cfg,
    configure_teacher_agent, configure_student_agent,
)
from legged_lab.utils.env_utils.fixed_bar_attachment import FixedBarAttachmentCfg
from legged_lab.terrains import FLAT_TERRAINS_CFG


@configclass
class SFYGRewards:
    track_velocity = RewardTermCfg(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.8, params={"std": .5})
    track_yaw = RewardTermCfg(func=mdp.track_ang_vel_z_world_exp, weight=1., params={"std": .5})
    vertical_velocity = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.)
    angular_velocity = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-.05)
    orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.)
    energy = RewardTermCfg(func=mdp.energy, weight=-.001)
    acceleration = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-.01)
    joint_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.)
    velocity_limits = RewardTermCfg(func=mdp.joint_vel_limits, weight=-.5, params={"soft_ratio": .9})
    termination = RewardTermCfg(func=mdp.is_terminated, weight=-200.)
    height = RewardTermCfg(func=mdp.loose_robot_height_reward, weight=1.5,
                           params={"std": .25, "deadband": .02, "asset_cfg": SceneEntityCfg("robot")})
    foot_slide = RewardTermCfg(func=mdp.feet_slide, weight=-.25, params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_pitch_[LR]_Link"),
        "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_[LR]_Link"),
        "contact_force_threshold": 1.,
    })
    air_time = RewardTermCfg(func=mdp.feet_air_time_positive_biped, weight=.15, params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_pitch_[LR]_Link"),
        "threshold": .4, "moving_command_threshold": .1,
    })
    hand_position = RewardTermCfg(func=collaboration_mdp.virtual_palm_position_reward, weight=1., params={
        "asset_cfg": SceneEntityCfg("robot", body_names="arm6_Link"),
        "std": .25, "deadband": .02, "command_start": 4, "wrist_to_palm_position": (0., 0., .1358),
    })
    hand_orientation = RewardTermCfg(func=collaboration_mdp.virtual_palm_quaternion_reward, weight=1., params={
        "asset_cfg": SceneEntityCfg("robot", body_names="arm6_Link"),
        "std": .25, "deadband": .02, "command_start": 0, "wrist_to_palm_position": (0., 0., .1358),
    })


def _common(cfg, profile):
    cfg.scene.robot = make_sfyg_asset()
    cfg.scene.terrain_type = "generator"
    cfg.scene.terrain_generator = FLAT_TERRAINS_CFG
    cfg.scene.height_scanner.prim_body_name = "base_Link"
    cfg.scene.foot_sensor_pattern = "ankle_pitch_[LR]_Link"
    cfg.robot.single_arm = True
    cfg.robot.policy_joint_names = profile["policy_joint_names"]
    cfg.robot.action_scale = profile["action_scale"]
    cfg.robot.feet_body_names = profile["feet_body_names"]
    cfg.robot.terminate_contacts_body_names = [
        "^(?!(ankle_pitch_[LR]_Link|arm6_Link|gripper.*|gripper_pick)$).*$"
    ]
    cfg.reward = SFYGRewards()
    for term in (cfg.reward.hand_position, cfg.reward.hand_orientation):
        term.params["asset_cfg"].body_names = profile["ee_body"]
        term.params["wrist_to_palm_position"] = tuple(profile["ee_local_position"])
    cfg.commands.debug_vis = False
    cfg.pose_commands.debug_vis = False
    cfg.commands.heading_command = False
    cfg.commands.ranges.heading = None
    cfg.commands.ranges.lin_vel_x = (-.3, .5)
    cfg.commands.ranges.lin_vel_y = (-.2, .2)
    cfg.commands.ranges.ang_vel_z = (-.5, .5)
    height = profile["base_position"][2]
    cfg.commands.ranges.height = (height - .04, height + .04)
    cfg.normalization.clip_actions = 5.
    events = cfg.domain_rand.events
    events.add_base_mass.params["asset_cfg"].body_names = ["base_Link"]
    events.randomize_coms_without_inertia.params["asset_cfg"].body_names = ["base_Link"]
    events.reset_body_link_mass.params["asset_cfg"].body_names = ["proximal_pitch_[LR]_Link"]
    events.reset_hand_mass = None
    events.randomize_joint_gains.params["asset_cfg"] = SceneEntityCfg("robot")
    # Keep reset exactly at the FK-calibrated grasp, including yaw pi and grippers.
    events.reset_robot_joints.params["position_range"] = (1., 1.)
    events.push_robot.params["velocity_range"] = {"x": (-.3, .3), "y": (-.3, .3)}


def make_configs(phase, no_object=False):
    if phase not in (1, 2, 3):
        raise ValueError("phase must be 1, 2 or 3")
    p = load_profile()
    xyz, quat = tuple(p["ee_position_base"]), tuple(p["ee_quaternion_base"])
    height = p["base_position"][2]
    if phase == 1:
        cfg = LocomotionEnvCfg()
        cfg.experiment = LocomotionExperimentCfg(
            termination_observation=TerminationObservationCfg(
                robot_contact_force_threshold=1., feet_contact_force_threshold=.5, object_min_height=None),
            pose_trajectory=LocomotionPoseTrajectoryCfg(
                num_waypoints=10, lateral_randomization_scale=1., vertical_randomization_scale=1.,
                left_center_xyz=xyz, right_center_xyz=xyz, left_center_quat=quat, right_center_quat=quat),
        )
        _common(cfg, p)
        cfg.pose_commands.ranges.cube_size = .03
        cfg.pose_commands.ranges.orientation_cone_rad = .15
        agent = Phase1LocomotionAgentCfg()
        name = "tron2_sfyg_phase_1_locomotion"
    else:
        cfg = CollaborationEnvCfg()
        cfg.robot = teacher_robot_cfg() if phase == 2 else student_robot_cfg()
        cfg.experiment = collaboration_experiment_cfg(
            masked_height_command=height if phase == 3 else None,
            vertical_velocity_noise=.05 if phase == 3 else 0.,
            student_mass_observation_enabled=phase == 3,
        )
        cfg.experiment.grasp.left_palm_xyz = xyz
        cfg.experiment.grasp.right_palm_xyz = xyz
        cfg.experiment.grasp.left_palm_quat = quat
        cfg.experiment.grasp.right_palm_quat = quat
        cfg.experiment.support_motion.no_object_height_command = height
        cfg.experiment.observations.student_no_object_true_mass_range_kg = tuple(p["bar_mass_range"])
        cfg.experiment.termination_observation.object_min_height = .25
        cfg.support_height_control = support_height_control_cfg()
        cfg.support_height_control.ankle_height_offset = 0.
        cfg.support_height_control.corrected_height_range = (.5, 2.5)
        cfg.bar_controller = bar_controller_cfg()
        length = p["bar_length"]
        cfg.bar_controller.bar_size = (.025, length, .025)
        cfg.bar_controller.bar_mass = sum(p["bar_mass_range"]) / 2
        center = (xyz[0], xyz[1] - length / 2, height + xyz[2])
        cfg.bar_controller.initial_center_position = center
        cfg.bar_controller.human_endpoint_offset = (0., -length / 2, 0.)
        cfg.bar_controller.robot_endpoint_offset = (0., length / 2, 0.)
        _common(cfg, p)
        body_quat = p["ee_body_quaternion_base"]
        cfg.scene.bar_attachment = FixedBarAttachmentCfg(
            secondary_attachment=False, fixed_side="left", left_wrist_body_name=p["ee_body"],
            left_wrist_position=tuple(p["ee_local_position"]),
            left_wrist_rotation=(body_quat[0], -body_quat[1], -body_quat[2], -body_quat[3]),
            left_bar_position=cfg.bar_controller.robot_endpoint_offset,
        )
        cfg.commands.ranges.height = (height, height)
        cfg.commands.ranges.link_height = (center[2] - .05, center[2] + .05)
        cfg.experiment.support_motion.bar_height_command_range = cfg.commands.ranges.link_height
        cfg.experiment.support_motion.bar_height_command_offset = 0.
        cfg.reward.track_velocity.func = mdp.track_lin_vel_xy_yaw_frame_exp_world
        events = cfg.domain_rand.events
        events.reset_root_and_box_link_setting_with_joint_state_uniform.func = collaboration_mdp.reset_robot_and_bar_uniform
        events.reset_root_and_box_link_setting_with_joint_state_uniform.params["velocity_range"] = {}
        if no_object:
            events.reset_target_object_mass = None
            events.reset_root_and_box_link_setting_with_joint_state_uniform = EventTermCfg(
                func=mdp.reset_root_state_uniform, mode="reset",
                params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}},
            )
            cfg.commands.rel_standing_envs = 1.
            cfg.commands.ranges.lin_vel_x = (0., 0.)
            cfg.commands.ranges.lin_vel_y = (0., 0.)
            cfg.commands.ranges.ang_vel_z = (0., 0.)
        else:
            events.reset_target_object_mass.params.update(
                asset_cfg=SceneEntityCfg("carried_bar"), mass_distribution_params=tuple(p["bar_mass_range"]),
                operation="abs", distribution="uniform", recompute_inertia=True,
            )
            cfg.reward.bar_height = RewardTermCfg(func=collaboration_mdp.bar_center_height_tracking_reward, weight=1., params={"std": .10})
            cfg.reward.bar_velocity = RewardTermCfg(func=collaboration_mdp.bar_center_horizontal_velocity_tracking_reward, weight=1., params={"std": .5})
            cfg.reward.bar_alignment = RewardTermCfg(func=collaboration_mdp.target_bar_vector_alignment_reward, weight=3., params={"epsilon": 1e-8})
            cfg.reward.bar_tilt = RewardTermCfg(func=collaboration_mdp.bar_endpoint_height_difference_penalty, weight=-10.)
            cfg.reward.human_effort = RewardTermCfg(func=collaboration_mdp.human_effort_reward, weight=1., params={"force_scale": 100., "epsilon": 1e-8})
        agent = TeacherAgentCfg() if phase == 2 else StudentAgentCfg()
        (configure_teacher_agent if phase == 2 else configure_student_agent)(agent)
        if phase == 2:
            agent.policy.class_name = "ActorCriticWbcEnd2endFollowingWholePipeQuatResidual"
        if phase == 3:
            agent.policy.teacher_action_clip = 5.
        name = f"tron2_sfyg_phase_{phase}_{'teacher' if phase == 2 else 'student'}_{'no_object' if no_object else 'fixed_bar'}"
    agent.experiment_name = name
    agent.wandb_project = "cola_tron2_sfyg"
    return cfg, agent
