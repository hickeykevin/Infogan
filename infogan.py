from utils import Config
import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor as tensor
import torch
from torch.autograd import Variable
from models import Generator, Discriminator
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np
import itertools
from datasets import MNISTDataModule as mnist



class LitInfoGAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.generator = Generator(opt=self.config)
        self.discriminator = Discriminator(opt=self.config)
        self.automatic_optimization = False

    # ----------------------
    # Helper functions
    # ----------------------
    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        return m  
    
    def to_categorical(self, y):
        """Returns one-hot encoded Variable"""
        y_cat = torch.zeros((y.shape[0], self.config.n_classes), device=self.device, dtype=torch.float)
        y_cat[range(y.shape[0]), y] = 1.0

        return y_cat

    def sample_generator_input(self, batch_size=32):
        z = torch.randn(size=(batch_size, self.config.latent_dim), device=self.device)
        label_input = self.to_categorical(
            torch.randint(low=0, high=self.config.n_classes, size=(batch_size,)), 
            )
        code_input = torch.as_tensor(np.random.uniform(-1, 1, (batch_size, self.config.code_dim)), dtype=torch.float, device=self.device)
        return z, label_input, code_input
    
    def sample_fixed_input(self):
        z = torch.randn(size=(self.config.n_classes ** 2, self.config.latent_dim), device=self.device)
        static_z = torch.zeros((self.config.n_classes ** 2, self.config.latent_dim), device=self.device, dtype=torch.float)
        
        static_label = self.to_categorical(
            torch.as_tensor(
                np.array([num for _ in range(self.config.n_classes) for num in range(self.config.n_classes)]), 
                ) 
            )
        
        static_code = torch.zeros((self.config.n_classes ** 2, self.config.code_dim), device=self.device, dtype=torch.float)
        return z, static_z, static_label, static_code

    def sample_varied_codes(self):
        # Get varied c1 and c2
        zeros = np.zeros((10 ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, self.config.n_classes)[:, np.newaxis], self.config.n_classes, 0)
        c1 = torch.as_tensor(np.concatenate((c_varied, zeros), -1), device=self.device, dtype=torch.float)
        c2 = torch.as_tensor(np.concatenate((zeros, c_varied), -1), device=self.device, dtype=torch.float)
        return c1, c2

    def on_fit_start(self):
        # Initialize generator and discriminator weights
        self.generator = self.weights_init_normal(self.generator)
        self.discriminator = self.weights_init_normal(self.discriminator)

    # ----------------------
    # Training Logic
    # ----------------------
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.shape[0]
        g_opt, d_opt, info_opt = self.optimizers()
        z, label_input, code_input = self.sample_generator_input(batch_size=batch_size)
        
        # -------------------------
        # Adversarial losses
        # -------------------------

        # Ground Truth Labels
        valid = tensor((batch_size, 1)).fill_(1.0).type_as(imgs)
        fake = tensor((batch_size, 1)).fill_(0.0).type_as(imgs)
        gen_imgs = self.generator(z, label_input, code_input)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        d_opt.zero_grad()
        
        # Loss for real images
        real_pred, _, _ = self.discriminator(imgs)
        d_real_loss = nn.functional.mse_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = self.discriminator(gen_imgs.detach())
        d_fake_loss = nn.functional.mse_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        self.manual_backward(d_loss)
        d_opt.step()

        # -------------------------
        # Train Generator
        # -------------------------
        
        g_opt.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label, pred_code = self.discriminator(gen_imgs)
        g_loss = nn.functional.mse_loss(validity, valid)
        self.manual_backward(g_loss)
        g_opt.step()
        self.log_dict({"g_loss": g_loss, "d_loss": d_loss})


        # ------------------
        # Information Loss
        # ------------------

        info_opt.zero_grad()

        # Sample generator input
        #z, label_input, code_input = self.sample_generator_input(batch_size=batch_size)

        #gen_imgs = self.generator(z, label_input, code_input)

        # Get discriminator output
        #_, pred_label, pred_code = self.discriminator(gen_imgs)

        # Sample ground truth labels
        gt_labels = torch.randint(low=0, high=self.config.n_classes, size=(batch_size,), device=self.device, dtype=torch.long)
        
        info_loss = self.config.lambda_cat * nn.functional.cross_entropy(pred_label, gt_labels) + self.config.lambda_con * nn.functional.mse_loss(
            pred_code, code_input
        )

        self.manual_backward(info_loss)
        info_opt.step()


    def configure_optimizers(self):
        optimizer_G = Adam(self.generator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))
        optimizer_D = Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))
        optimizer_info = Adam(
            itertools.chain(self.generator.parameters(), self.discriminator.parameters()), lr=self.config.lr, betas=(self.config.b1, self.config.b2)
        )
        return optimizer_G, optimizer_D, optimizer_info
