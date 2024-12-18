# trunk-ignore(bandit/B403)
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DaphnetDataset(Dataset):
    def __init__(self, config, mode="train", strategy_type=None):
        self.config = config
        self.root = config["data"]["root"]
        self.mode = mode
        self.strategy_type = strategy_type

        # trunk-ignore(bandit/B301)
        self.data = pickle.load(open(f"{self.root}/daphnet_{self.mode}.pkl", "rb"))
        self.label = np.zeros((self.data.shape[0]))
        if self.mode == "test":
            # trunk-ignore(bandit/B301)
            self.label = pickle.load(
                open(f"{self.root}/daphnet_{self.mode}_label.pkl", "rb")
            )
        self.data = torch.Tensor(self.data)
        self.label = torch.LongTensor(self.label)

        self.window_length = config["data"]["window_length"]
        self.begin_indices = None
        if config["data"]["overlap"]:
            self.begin_indices = list(range(0, len(self.data) - self.window_length))
        if not config["data"]["overlap"]:
            self.begin_indices = list(range(0, len(self.data), self.window_length))

        self.segments = config["data"]["segments"]

    def create_mask(self, observed_mask, strategy_type):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        if strategy_type == 0:
            skip = length // self.segments
            for segment_idx, begin_index in enumerate(list(range(0, length, skip))):
                if segment_idx % 2 == 0:
                    mask[begin_index : min(begin_index + skip, length), :] = 1
        else:
            skip = length // self.segments
            for segment_idx, begin_index in enumerate(list(range(0, length, skip))):
                if segment_idx % 2 != 0:
                    mask[begin_index : min(begin_index + skip, length), :] = 1
        return mask

    def __len__(self):
        return len(self.begin_indices)

    def __getitem__(self, idx):
        if self.mode == "train":
            # trunk-ignore(bandit/B311)
            if random.random() < 0.5:
                strategy_type = 0
            else:
                strategy_type = 1
        elif self.mode == "test":
            strategy_type = self.strategy_type

        observed_data = self.data[
            self.begin_indices[idx] : self.begin_indices[idx] + self.window_length
        ]
        observed_mask = torch.ones_like(observed_data)
        gt_mask = self.create_mask(observed_mask, strategy_type)
        timepoints = np.arange(self.window_length)
        label = self.label[
            self.begin_indices[idx] : self.begin_indices[idx] + self.window_length
        ]

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "strategy_type": strategy_type,
            "timepoints": timepoints,
            "label": label,
        }


def get_dataloader(config):
    train_ds = DaphnetDataset(config, mode="train")
    train_ds, val_ds = random_split(
        train_ds,
        [len(train_ds) - int(0.05 * len(train_ds)), int(0.05 * len(train_ds))],
    )
    test_0_ds = DaphnetDataset(config, mode="test", strategy_type=0)
    test_1_ds = DaphnetDataset(config, mode="test", strategy_type=1)

    train_dl = DataLoader(
        train_ds, batch_size=config["data"]["batch_size"], shuffle=True
    )
    val_dl = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=True)
    test_0_dl = DataLoader(
        test_0_ds, batch_size=config["data"]["batch_size"], shuffle=False
    )
    test_1_dl = DataLoader(
        test_1_ds, batch_size=config["data"]["batch_size"], shuffle=False
    )

    return train_dl, val_dl, test_0_dl, test_1_dl


if __name__ == "__main__":
    import json

    path = "configs/config_example.json"
    with open(path, "r") as f:
        config = json.load(f)

    dataset = DaphnetDataset(config, mode="train")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        print(data)
        break
