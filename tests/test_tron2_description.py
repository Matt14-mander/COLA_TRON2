import tempfile
import unittest
import json
from pathlib import Path

from legged_lab.utils.tron2_description import inspect_variant
from legged_lab.assets.tron2 import make_tron2_asset


ROOT = Path(__file__).resolve().parents[1]


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.variant = self.root / "tron2" / "TEST"
        for folder in ("urdf", "meshes", "usd"):
            (self.variant / folder).mkdir(parents=True)
        (self.variant / "meshes" / "body.stl").touch()
        (self.variant / "usd" / "robot.usd").touch()
        (self.variant / "urdf" / "robot.urdf").write_text('''<robot name="test">
          <link name="base"><visual><geometry><mesh filename="package://legacy/meshes/body.stl"/></geometry></visual></link>
          <link name="foot"/><joint name="hip" type="revolute">
          <parent link="base"/><child link="foot"/>
          <limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
          <joint name="fixed" type="fixed"><parent link="base"/><child link="foot"/></joint>
        </robot>''', encoding="utf-8")

    def test_inventory_and_package_alias(self):
        report = inspect_variant(self.root, "TEST")
        self.assertEqual(report["urdf_joint_order"], ["hip"])
        self.assertEqual(report["missing_meshes"], [])
        self.assertEqual(report["required_ros_package_aliases"], {"legacy": str(self.variant)})
        self.assertEqual(report["joints"][0]["limits"]["effort"], 10)
        self.assertFalse(report["training_ready"])

    def test_missing_mesh(self):
        (self.variant / "meshes" / "body.stl").unlink()
        self.assertEqual(len(inspect_variant(self.root, "TEST")["missing_meshes"]), 1)

    def test_reject_unknown_variant(self):
        with self.assertRaises(ValueError):
            inspect_variant(self.root, "../TEST")

    def test_asset_requires_real_parameters_before_importing_simulator(self):
        with self.assertRaises(ValueError):
            make_tron2_asset(self.root, "TEST", joint_parameters={}, base_position=(0, 0, 1))
        values = dict(default_position=2, stiffness=10, damping=1, effort_limit=10, velocity_limit=2)
        with self.assertRaisesRegex(ValueError, "outside URDF"):
            make_tron2_asset(self.root, "TEST", joint_parameters={"hip": values}, base_position=(0, 0, 1))

    def test_bundled_sfyg_profile_matches_robot_description(self):
        profile = json.loads((ROOT / "assets/tron2/sfyg_profile.json").read_text(encoding="utf-8"))
        report = inspect_variant(ROOT / "assets/tron2", profile["variant"])
        parameters = profile["joint_parameters"]
        self.assertEqual(report["movable_joint_count"], 18)
        self.assertEqual(set(parameters), set(report["urdf_joint_order"]))
        self.assertEqual(len(profile["policy_joint_names"]), 16)
        self.assertEqual(
            set(parameters) - set(profile["policy_joint_names"]),
            {"gripper1_Joint", "gripper2_Joint"},
        )
        self.assertEqual(set(profile["feet_body_names"]), {"ankle_pitch_L_Link", "ankle_pitch_R_Link"})
        self.assertIn(profile["ee_body"], report["body_names"])
        limits = {joint["name"]: joint["limits"] for joint in report["joints"]}
        for name, values in parameters.items():
            self.assertLessEqual(limits[name]["lower"], values["default_position"])
            self.assertLessEqual(values["default_position"], limits[name]["upper"])


if __name__ == "__main__":
    unittest.main()
