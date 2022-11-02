import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import sequence_generator
from typing import Optional
import timeit

# Dataset: Spits out sequence with whether possible_terms exist


class FunctionDataset(Dataset):
    def __init__(
        self, size=100, generate_on_call=False, additional_function_maker_params=()
    ) -> None:
        """If generate on call, we randomly make a function every time this thing
        is called, which will probably give different functions across
        different epochs.

        Otherwise, generate all the functions at init.

        Args:
            size (int, optional): _description_. Defaults to 1000.
            generate_on_call (bool, optional): _description_. Defaults to False.
            additional_function_maker_params (tuple, optional): _description_. Defaults to ().
        """

        super().__init__()
        self.size = size
        self.additional_function_maker_params = additional_function_maker_params
        # self.num_possible_terms = len(sequence_generator.make_possible_terms())
        # self.possible_functions = sequence_generator.make_possible_functions()
        self.generate_on_call = generate_on_call
        self.data = []

        if not self.generate_on_call:
            self.data = sequence_generator.make_n_random_functions(
                self.size, torchify=True
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        if self.generate_on_call:
            # We don't need to use index here because everything is randomly generated
            f, seq, terms = sequence_generator.make_random_function()
            seq = torch.tensor(seq, dtype=torch.float)
            terms = torch.tensor(terms, dtype=torch.float)
            return seq, terms
        else:
            return self.data[idx]


class FunctionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = FunctionDataset(size=3000)
        self.val_set = FunctionDataset(size=300)
        self.test_set = FunctionDataset(size=300)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    start = timeit.default_timer()
    f = FunctionDataset(size=10000)
    end = timeit.default_timer()
    print("Time elapsed", end - start)
    print(f[0])
