from .base import BaseRobot
from pathlib import Path

from ..dataset.dataset import Dataset


class SimRobot(BaseRobot):
    """Simulates a robot based on a dataset.

    Args:
        dataset_path: Path to the dataset.
    """

    def __init__(self, dataset_path: Path):
        self.dataset = Dataset(dataset_path)

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self):

        return {"gripper.pos": 0}

    def send_action(self, action: dict):
        pass
