from collections import defaultdict
from typing import Dict, Iterable, List

from datasets import load_dataset


class Dataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = load_dataset(self.dataset_path)
        self.train_split = self.dataset["train"]
        self.sequence_num = 0
        self._episode_indices = self._split_by_episode_index()

    def _split_by_episode_index(self) -> Dict[int, List[int]]:
        """Group dataset indices by episode_index for quick lookups."""
        grouped_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(self.train_split)):
            record = self.train_split[idx]
            episode_idx = record.get("episode_index")
            if episode_idx is None:
                raise KeyError(
                    "Record is missing the 'episode_index' key, cannot split dataset."
                )
            grouped_indices[int(episode_idx)].append(idx)
        # Convert back to a plain dict so downstream users can inspect / iterate.
        return dict(sorted(grouped_indices.items()))

    @property
    def num_episodes(self) -> int:
        return len(self._episode_indices)

    def get_episode_indices(self, episode_index: int) -> List[int]:
        if episode_index not in self._episode_indices:
            raise IndexError(
                f"Episode {episode_index} not found. Available episodes: {list(self._episode_indices)}"
            )
        return self._episode_indices[episode_index]

    def get_episode(self, episode_index: int) -> List[dict]:
        """Return all records that belong to a single episode."""
        return [self.train_split[i] for i in self.get_episode_indices(episode_index)]

    def iter_episode(self, episode_index: int) -> Iterable[dict]:
        """Yield records in a streaming fashion for the provided episode."""
        for idx in self.get_episode_indices(episode_index):
            yield self.train_split[idx]

    def get_observation(self):
        return self.train_split[self.sequence_num]


if __name__ == "__main__":
    dataset = Dataset("abemii/record-test-3-e4")
    print(f"Found {dataset.num_episodes} episodes.")

