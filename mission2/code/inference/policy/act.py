from pathlib import Path

import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline


class ActPolicy:
    def __init__(self, pretrained_path: Path):
        self.policy = ACTPolicy.from_pretrained(pretrained_path).cuda()
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="policy_preprocessor.json"
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="policy_postprocessor.json"
        )

    def inference(self, observation: dict) -> torch.Tensor:
        batched_observation = self.preprocessor(observation)
        action = self.policy.select_action(batched_observation)
        post_processed = self.postprocessor({"action": action})
        return post_processed["action"].squeeze(0).to(dtype=torch.float32)
