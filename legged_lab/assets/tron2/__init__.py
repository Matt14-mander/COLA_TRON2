"""TRON2 asset construction; no implicit choice of hardware variant."""


def make_tron2_asset(description_root, variant, *, joint_parameters, base_position):
    """Build a position-controlled asset using explicitly supplied nominal gains.

    joint_parameters maps every movable URDF joint to default_position, stiffness,
    damping, effort_limit and velocity_limit (SI units). Explicit effort/velocity
    values are required because source URDF and MJCF limits may differ. Wheel
    joints need a separate velocity-control adapter and are rejected here.
    """
    import math

    from legged_lab.utils.tron2_description import inspect_variant

    report = inspect_variant(description_root, variant)
    if not report["usd_exists"]:
        raise FileNotFoundError(report["usd"])
    names = report["urdf_joint_order"]
    if set(joint_parameters) != set(names):
        raise ValueError("joint_parameters must cover exactly the movable URDF joints")
    if any("wheel" in name.lower() for name in names):
        raise ValueError("Wheeled TRON2 needs velocity-control actions; position-only adapter is unsupported")
    if len(base_position) != 3 or not all(math.isfinite(v) for v in base_position):
        raise ValueError("base_position must be three finite SI values")
    fields = ("default_position", "stiffness", "damping", "effort_limit", "velocity_limit")
    for joint in report["joints"]:
        name = joint["name"]
        values = joint_parameters[name]
        if any(key not in values or not math.isfinite(values[key]) for key in fields):
            raise ValueError(f"Missing or invalid actuator parameters for {name}")
        if any(values[key] <= 0 for key in ("stiffness", "effort_limit", "velocity_limit")) or values["damping"] < 0:
            raise ValueError(f"Invalid actuator limits/gains for {name}")
        limits = joint["limits"]
        if joint["type"] != "continuous" and not (
            limits.get("lower", -math.inf) <= values["default_position"] <= limits.get("upper", math.inf)
        ):
            raise ValueError(f"Default position outside URDF limits: {name}")

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=report["usd"],
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(base_position),
            joint_pos={name: joint_parameters[name]["default_position"] for name in names},
            joint_vel={".*": 0.0},
        ),
        actuators={
            name: ImplicitActuatorCfg(
                joint_names_expr=[name],
                stiffness=joint_parameters[name]["stiffness"],
                damping=joint_parameters[name]["damping"],
                effort_limit_sim=joint_parameters[name]["effort_limit"],
                velocity_limit_sim=joint_parameters[name]["velocity_limit"],
            )
            for name in names
        },
    )
