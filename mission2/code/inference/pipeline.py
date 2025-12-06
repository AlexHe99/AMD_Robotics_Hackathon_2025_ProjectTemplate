from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import make_default_processors
from .robot.so101 import So101MotorPosNames
from .robot.base import BaseRobot
from .policy.act import ActPolicy
import torch


def _postprocess_action(action: torch.Tensor) -> dict:
    assert action.shape == (6,)
    return {
        So101MotorPosNames.SHOULDER_PAN.value: action[0],
        So101MotorPosNames.SHOULDER_LIFT.value: action[1],
        So101MotorPosNames.ELBOW_FLEX.value: action[2],
        So101MotorPosNames.WRIST_FLEX.value: action[3],
        So101MotorPosNames.WRIST_ROLL.value: action[4],
        So101MotorPosNames.GRIPPER.value: action[5],
    }


class InferencePipeline(object):
    def __init__(self, robot: BaseRobot, policy: ActPolicy):
        self.robot = robot
        self.policy = policy
        (_, _, self.robot_observation_processor) = make_default_processors()

    def run(self):
        for _ in range(1000):
            observation_raw = self.robot.get_observation()
            observation = prepare_observation_for_inference(observation_raw, "cuda")
            action = self.policy.inference(observation)
            action_dict = _postprocess_action(action)
            self.robot.send_action(action_dict)
