import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import dataset

class TermPredictor(pl.LightningModule):
    # current version: Z^10 -> F_2^10
    # represented as floats
    def __init__(self, input_length = 10, output_length = 10) -> None:
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length

        self.a1 = torch.nn.MultiheadAttention(3*self.input_length, 1)
        self.a2 = torch.nn.MultiheadAttention(3*self.input_length, 1)
        self.a3 = torch.nn.MultiheadAttention(3*self.input_length, 1)
        self.fc = torch.nn.Linear(3*self.input_length, output_length)

    def safe_log(self, x, eps=1e-7):
        # so that log doesn't go to 0 when applied twice
        x = F.relu(x)
        x = torch.log(x + eps)
        return x

    def forward(self, x):
        # Concatenate x with log(x) and log(log(x))
        # TODO: fix this
        augmented_tensor = torch.zeros(self.input_length* 3)
        augmented_tensor[0 : self.input_length] = x
        augmented_tensor[self.input_length : self.input_length * 2] = self.safe_log(x)
        augmented_tensor[self.input_length * 2: self.input_length * 3] = self.safe_log(self.safe_log(x))
        augmented_tensor = augmented_tensor[None, :]

        augmented_tensor = self.a1(augmented_tensor, augmented_tensor, augmented_tensor)[0]
        augmented_tensor = self.a2(augmented_tensor, augmented_tensor, augmented_tensor)[0]
        augmented_tensor = self.a3(augmented_tensor, augmented_tensor, augmented_tensor)[0]

        augmented_tensor = augmented_tensor[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

if __name__ == '__main__':

    # tp = TermPredictor()
    # fd = dataset.FunctionDataset(size=1)
    # i = fd[0]
    # x, y = i
    # tp.forward(x[None, :])

    model = TermPredictor()
    dm = dataset.FunctionDataModule()

    trainer = pl.Trainer()
    trainer.fit(model, datamodule=dm)