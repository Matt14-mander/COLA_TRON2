"""Read TRON2 model contracts without importing Isaac Sim or torch."""

from pathlib import Path
import xml.etree.ElementTree as ET


def inspect_variant(description_root, variant):
    root = Path(description_root).resolve()
    if variant not in {p.name for p in (root / "tron2").iterdir() if p.is_dir()}:
        raise ValueError(f"Unknown TRON2 variant: {variant}")
    folder = root / "tron2" / variant
    urdf = folder / "urdf" / "robot.urdf"
    model = ET.parse(urdf).getroot()
    joints = []
    for joint in model.findall("joint"):
        if joint.get("type") in ("fixed", "floating"):
            continue
        limit = joint.find("limit")
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "limits": {} if limit is None else {k: float(v) for k, v in limit.attrib.items()},
        })
    names = [j["name"] for j in joints]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate joints in {urdf}")
    missing_meshes = []
    package_aliases = {}
    for mesh in model.findall(".//mesh"):
        reference = mesh.get("filename", "")
        if reference.startswith("package://"):
            relative = reference[len("package://"):].split("/", 1)
            path = root / relative[-1]
            if not path.is_file() and (folder / relative[-1]).is_file():
                path = folder / relative[-1]
                package_aliases[relative[0]] = str(folder)
        else:
            path = urdf.parent / reference
        if not path.is_file():
            missing_meshes.append(reference)
    return {
        "variant": variant,
        "urdf": str(urdf),
        "usd": str(folder / "usd" / "robot.usd"),
        "usd_exists": (folder / "usd" / "robot.usd").is_file(),
        "mjcf": str(folder / "xml" / "robot.xml"),
        "mjcf_exists": (folder / "xml" / "robot.xml").is_file(),
        "movable_joint_count": len(joints),
        "urdf_joint_order": names,
        "joints": joints,
        "body_names": [link.get("name") for link in model.findall("link")],
        "missing_meshes": sorted(set(missing_meshes)),
        "required_ros_package_aliases": package_aliases,
        "training_ready": False,
        "required_configuration": [
            "Selected hardware variant and policy-controlled joints",
            "PD gains, default pose, action scales and control modes",
            "Feet/end-effector frames and task topology",
            "TRON2 observation/reward configuration and compatible rsl_rl networks",
        ],
    }
