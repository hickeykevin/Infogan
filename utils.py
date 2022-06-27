from dataclasses import dataclass
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

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
  sample_interval: wandint=400
  lambda_cat: int=1
  lambda_con: float = 0.1

LOGGER = WandbLogger(project="infogan")

class LogGeneratedImagesCallback(Callback):
  def __init__(self):
    super().__init__()

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        if batch_idx in [600, 1200, 1700]:
          
          z, static_z, static_label, static_code = pl_module.sample_fixed_input()
          # generate static codes using fixed generator inputs
          gen_imgs_static = pl_module.generator(z, static_label, static_code)
          LOGGER.log_image(key="static_generated_images", images=[x for x in gen_imgs_static])

          #generate varied codes
          c1_varied, c2_varied = pl_module.sample_varied_codes()
          gen_imgs_c1_varied = pl_module.generator(static_z, static_label, c1_varied)
          gen_imgs_c2_varied = pl_module.generator(static_z, static_label, c2_varied)
          LOGGER.log_image(key="c1_varied_generated_images", images=[x for x in gen_imgs_c1_varied])
          LOGGER.log_image(key="c2_varied_generated_images", images=[x for x in gen_imgs_c2_varied])