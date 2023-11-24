import lightning as L
from dataset import MNISTDataModule
import config
from GAN import GAN
import torch
torch.set_float32_matmul_precision("medium") # to make lightning happy
if __name__ == "__main__":
    dm = MNISTDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = GAN(*dm.dims)
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        max_epochs=config.MAX_EPOCHS,
    )
    trainer.fit(model, dm)

