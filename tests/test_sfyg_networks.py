"""CPU tensor and checkpoint contracts; requires the project's rsl_rl submodule."""
import unittest
import torch
from rsl_rl.modules import ActorCriticWbcEnd2endQuat, ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29, StudentTeacherDistill
from legged_lab.utils.joint_mapping import position_targets


class SFYGNetworkTests(unittest.TestCase):
    def test_action_scatter_preserves_gripper(self):
        defaults = torch.tensor([[.025, .2, .025, -.4]])
        actual = position_targets(defaults, torch.tensor([[1., -1.]]), [3, 1], .1)
        torch.testing.assert_close(actual, torch.tensor([[.025, .1, .025, -.3]]))
        torch.testing.assert_close(defaults, torch.tensor([[.025, .2, .025, -.4]]))

    def test_three_phase_checkpoints_and_tail_removal(self):
        base = ActorCriticWbcEnd2endQuat(650, 700, 16, actor_hidden_dims=[16], critic_hidden_dims=[16], device="cpu")
        teacher = ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29(
            850, 900, 16, actor_hidden_dims=[16], critic_hidden_dims=[16],
            residual_hidden_init_std=.01, residual_final_init_std=0., residual_bias_init=0.,
            base_privileged_obs_per_frame=20, device="cpu",
        )
        self.assertFalse(teacher.load_state_dict(base.state_dict()))
        frames = torch.randn(2, 10, 85)
        base_input, residual = teacher.actor_forward(frames.flatten(1))
        torch.testing.assert_close(base_input, frames[:, :, :65].flatten(1))
        torch.testing.assert_close(residual, torch.zeros(2, 16))
        critic_frames = torch.randn(2, 10, 90)
        critic_input, _ = teacher.critic_forward(critic_frames.flatten(1))
        torch.testing.assert_close(critic_input, torch.cat((critic_frames[:, :, :65], critic_frames[:, :, -5:]), -1).flatten(1))
        student = StudentTeacherDistill(1650, 850, 16, student_hidden_dims=[16], teacher_hidden_dims=[16],
                                       teacher_base_privileged_obs_per_frame=20, teacher_action_clip=5.)
        self.assertFalse(student.load_state_dict(teacher.state_dict()))
        torch.testing.assert_close(student.evaluate(frames.flatten(1)), teacher.act_inference(frames.flatten(1)).clamp(-5, 5))
        self.assertEqual(student.act_inference(torch.zeros(2, 1650)).shape, (2, 16))
        self.assertTrue(student.load_state_dict(student.state_dict()))
        base.actor_cnn_writer.close()
        base.critic_cnn_writer.close()
        teacher.residual_writer.close()


if __name__ == "__main__":
    unittest.main()
