import unittest

from legged_lab.utils.joint_mapping import resolve_policy_joints, position_targets


class JointMappingTests(unittest.TestCase):
    def test_explicit_policy_order(self):
        self.assertEqual(resolve_policy_joints(["finger", "knee", "hip"], ["hip", "knee"]), [2, 1])

    def test_invalid_maps(self):
        for policy in ([], ["hip", "hip"], ["absent"]):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                resolve_policy_joints(["hip"], policy)

    def test_no_implicit_tron2_order(self):
        with self.assertRaises(ValueError):
            resolve_policy_joints(["joint"] * 10)
        with self.assertRaises(ValueError):
            resolve_policy_joints([f"joint{i}" for i in range(10)])

    def test_g1_compatibility(self):
        names = [f"body{i}" for i in range(29)] + ["left_hand_joint"]
        self.assertEqual(resolve_policy_joints(names), list(range(29)))


if __name__ == "__main__":
    unittest.main()
