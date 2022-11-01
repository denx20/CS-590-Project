import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import sequence_generator
from typing import Optional

# Dataset: Spits out sequence with whether possible_terms exist
class FunctionDataset(Dataset):
    def __init__(self, size=128) -> None:
        super().__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> int:
        pass


# class FunctionDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size: int = 32):
#         super().__init__()
#         self.batch_size = batch_size

#     def setup(self, stage: Optional[str] = None) -> None:

if __name__ == '__main__':
    f = FunctionDataset()