# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Custom torch datasets
"""
import tqdm.auto as tqdm
from torch.utils.data.dataset import Dataset


class CachedDataset(Dataset):
    """
    Dataset which is entirely stored in memory. Useful for small validation datasets.
    """

    def __init__(self, original_dataset: Dataset):
        self.dataset = original_dataset
        self.data = None
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.data = [
            self._prepare_item(self.dataset[index])
            for index in tqdm.trange(len(self.dataset))
        ]

    def update_prepared_data(self):
        self.data = None  # free memory
        self._prepare_dataset()

    def _prepare_item(self, item):
        # method to override
        return item

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.dataset)
