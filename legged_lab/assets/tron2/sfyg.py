"""SFYG simulation profile. Gains are tunable simulation defaults, not calibration."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = ROOT / "assets" / "tron2" / "sfyg_profile.json"


def load_profile():
    path = Path(os.environ.get("COLA_TRON2_PROFILE", PROFILE_PATH))
    return json.loads(path.read_text(encoding="utf-8"))


def description_root():
    return Path(os.environ.get("COLA_TRON2_DESCRIPTION_ROOT", ROOT / "assets" / "tron2"))


def make_sfyg_asset():
    from legged_lab.assets.tron2 import make_tron2_asset

    profile = load_profile()
    return make_tron2_asset(
        description_root(), "SFYG_TRON2A",
        joint_parameters=profile["joint_parameters"],
        base_position=profile["base_position"],
    )
