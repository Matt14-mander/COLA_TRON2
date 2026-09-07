import tempfile
import unittest
from pathlib import Path

from legged_lab.utils.tron2_description import inspect_variant
from legged_lab.assets.tron2 import make_tron2_asset


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


if __name__ == "__main__":
    unittest.main()
