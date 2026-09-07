"""Simulator-independent policy joint ordering and full-articulation targets."""


def resolve_policy_joints(available_names, policy_names=None):
    """Resolve exact names in policy order; preserve the released G1 fallback."""
    available_names = list(available_names)
    if len(set(available_names)) != len(available_names):
        raise ValueError("Articulation contains duplicate joint names")
    if policy_names is None:
        if len(available_names) < 29 or any("hand" not in n for n in available_names[29:]):
            raise ValueError("Non-G1 assets must define robot.policy_joint_names explicitly")
        policy_names = available_names[:29]
    policy_names = list(policy_names)
    if not policy_names or len(set(policy_names)) != len(policy_names):
        raise ValueError("policy_joint_names must be nonempty and unique")
    missing = [name for name in policy_names if name not in available_names]
    if missing:
        raise ValueError(f"Policy joints missing from articulation: {missing}")
    return [available_names.index(name) for name in policy_names]


def position_targets(default_positions, actions, joint_ids, scale):
    """Scatter policy offsets; keep uncontrolled joints at their default pose."""
    if actions.shape[-1] != len(joint_ids):
        raise ValueError(f"Expected {len(joint_ids)} actions, received {actions.shape[-1]}")
    targets = default_positions.clone()
    targets[:, joint_ids] = default_positions[:, joint_ids] + actions * scale
    return targets
