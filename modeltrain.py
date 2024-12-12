import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import aStar
import data
import training


@hydra.main(config_path="Configurations", config_name="trainingConfig")
def main(config):

    baseDirectory = f"{config.logdir}/{os.path.basename(config.dataset)}"

    training.set_global_seeds(1234)
    trainLoader = data.create_dataloader("C:/Users/Rkail/PycharmProjects/MAE551_projTemplate/planning-datasets/data/mpd/mazes_032_moore_c8.npz",
                                          "train",
                                          100,
                                          shuffle=True)
    valLoader = data.create_dataloader("C:/Users/Rkail/PycharmProjects/MAE551_projTemplate/planning-datasets/data/mpd/mazes_032_moore_c8.npz",
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
    logDir = "model/mazes_032_moore_c8"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logDir,
        max_epochs=20,
        callbacks=[checkpointCallback],
    )
    trainer.fit(module,trainLoader, valLoader)

if __name__ == "__main__":
    main()