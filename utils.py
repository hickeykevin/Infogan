from dataclasses import dataclass
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch import from_numpy
import torch
import numpy as np

DATA_FOLDER = Path.cwd() / "data"

@dataclass
class Config:
  n_epochs: int=200
  batch_size: int=64
  lr: float=2e-4
  b1: float=0.5
  b2: float=0.99
  n_cpu: int=8
  latent_dim: int=62
  code_dim: int=2
  n_classes: int=10
  img_size: int=32
  channels: int=1
  sample_interval: int=400
  lambda_cat: int=1
  lambda_con: float = 0.1

LOGGER = WandbLogger(project="infogan")

class LogGeneratedImagesCallback(Callback):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        if batch_idx in [600, 1200, 1700]:
        #r = np.random.RandomState(1234)
        #z = r.randn(100, self.config.latent_dim)
        #static_z = r.randn(100, self.config.latent_dim)
        #static_label = self.to_categorical(
        #  r.randint(low=0, high=self.config.n_classes, size=(100, ))
        #)
        #static_code = r.uniform(-1, 1, size=(100, self.config.code_dim))
          z = torch.randn(size=(self.config.n_classes ** 2, self.config.latent_dim), device=pl_module.device)
          static_z = torch.zeros((self.config.n_classes ** 2, self.config.latent_dim), device=pl_module.device, dtype=torch.float)
        
          static_label = pl_module.to_categorical(
            torch.as_tensor(
                np.array([num for _ in range(self.config.n_classes) for num in range(self.config.n_classes)]), 
                ) 
            )
        
          static_code = torch.zeros((self.config.n_classes ** 2, self.config.code_dim), device=pl_module.device, dtype=torch.float)
          
          
          #static_z = from_numpy(static_z).float().to(pl_module.device)
          #static_label = static_label.float().to(pl_module.device)
          #static_code = from_numpy(static_code).float().to(pl_module.device)
          # generate static codes using fixed generator inputs
          gen_imgs_static = pl_module.generator(z, static_label, static_code)
          LOGGER.log_image(key="static_generated_images", images=[x for x in gen_imgs_static])

          #generate varied codes
          c1_varied, c2_varied = self.sample_varied_codes()
          gen_imgs_c1_varied = pl_module.generator(static_z, static_label, c1_varied.float().to(pl_module.device))
          gen_imgs_c2_varied = pl_module.generator(static_z, static_label, c2_varied.float().to(pl_module.device))
          LOGGER.log_image(key="c1_varied_generated_images", images=[x for x in gen_imgs_c1_varied])
          LOGGER.log_image(key="c2_varied_generated_images", images=[x for x in gen_imgs_c2_varied])
  

  def sample_varied_codes(self):
      # Get varied c1 and c2
      zeros = np.zeros((10 ** 2, 1))
      c_varied = np.repeat(np.linspace(-1, 1, self.config.n_classes)[:, np.newaxis], self.config.n_classes, 0)
      c1 = torch.from_numpy(np.concatenate((c_varied, zeros), -1))
      c2 = torch.from_numpy(np.concatenate((zeros, c_varied), -1))
      return c1, c2



  