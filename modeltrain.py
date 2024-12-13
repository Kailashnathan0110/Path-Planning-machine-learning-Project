import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import aStar
import data
import training
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt


@hydra.main(config_path="Configurations", config_name="trainingConfig")
def main(config):

    training_data_file = f"{config.trainingData}"

    training.set_global_seeds(1234)
    trainLoader = data.create_dataloader(f"{training_data_file}.npz",
                                          "train",
                                          100,
                                          shuffle=True)
    valLoader = data.create_dataloader(f"{training_data_file}.npz",
                                          "valid",
                                          100,
                                          shuffle=False)

    neuralAstar = aStar.NeuralAstar(encoder_input="m+",
                                    encoder_arch="CNN",
                                    encoder_depth=4,
                                    learn_obstacles=False,
                                    Tmax=0.25)

    checkpointCallback = ModelCheckpoint(monitor="metrics/h_mean",
                                         save_weights_only=True,
                                         mode="max")
    module = training.PlannerModule(neuralAstar,config)
    logDir = f"model/{training_data_file}"




    lfCallback = LossPlotCallback()
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logDir,
        max_epochs=20,
        callbacks=[checkpointCallback ,lfCallback],
    )
    trainer.fit(module,trainLoader, valLoader)
    lfCallback.plot_losses()

class LossPlotCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())
        elif "metrics/train_loss" in trainer.callback_metrics:  # Adjusted for your logger
            self.train_losses.append(trainer.callback_metrics["metrics/train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())
        elif "metrics/val_loss" in trainer.callback_metrics:  # Adjusted for your logger
            self.val_losses.append(trainer.callback_metrics["metrics/val_loss"].item())

    def plot_losses(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()
if __name__ == "__main__":
    main()