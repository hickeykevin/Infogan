from gc import callbacks
import pytorch_lightning as pl
from datasets import MNISTDataModule
from utils import DATA_FOLDER, Config, LogGeneratedImagesCallback, LOGGER
from infogan import LitInfoGAN
from models import Generator, Discriminator


dataset = MNISTDataModule(data_dir=str(DATA_FOLDER), config=Config)
model = LitInfoGAN(config=Config)
LOGGER.watch(model.generator)
LOGGER.watch(model.discriminator)

trainer = pl.Trainer(
    max_epochs=20, 
    gpus=1, 
    logger=LOGGER,
    callbacks=[LogGeneratedImagesCallback(Config)], 
    #fast_dev_run=True
    )

if __name__ == "__main__":
    trainer.fit(model, dataset)


