"""Training loops for the GAN architectures."""

import torch
import torch.nn as nn
from tqdm import tqdm
from src.training.losses import get_adversarial_loss, custom_adversarial_loss

class GANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        device,
        target_model=None,
        lambda_weight=0.1,
        generator_steps=1,
        discriminator_steps=1,
    ):
        """
        Initializes the GAN trainer.
        
        Args:
            generator (nn.Module): The generator model.
            discriminator (nn.Module): The discriminator model.
            g_optimizer (torch.optim.Optimizer): Optimizer for the generator.
            d_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.
            device (torch.device): Device to run training on (CPU or CUDA).
            target_model (nn.Module, optional): The victim model for custom loss.
            lambda_weight (float): Weight for the custom target model loss.
            generator_steps (int): Number of generator updates per batch.
            discriminator_steps (int): Number of discriminator updates per batch.
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.target_model = target_model.to(device) if target_model else None
        self.lambda_weight = lambda_weight
        self.generator_steps = int(generator_steps)
        self.discriminator_steps = int(discriminator_steps)
        if self.generator_steps < 1 or self.discriminator_steps < 1:
            raise ValueError("generator_steps and discriminator_steps must both be >= 1.")
        
        self.adversarial_loss = get_adversarial_loss().to(device)

    def _train_generator(self, batch_size):
        """
        Trains the generator for one step.
        """
        self.g_optimizer.zero_grad()

        # Sample noise as generator input
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)

        # Generate a batch of images
        gen_imgs = self.generator(z)

        # Loss measures generator's ability to fool the discriminator
        d_pred = self.discriminator(gen_imgs)
        g_adv_loss = self.adversarial_loss(d_pred, torch.ones_like(d_pred, device=self.device)) # We want D to think they are real (1)
        
        # Add custom target model loss if applicable
        g_target_loss = custom_adversarial_loss(gen_imgs, self.target_model, self.lambda_weight)
        
        g_loss = g_adv_loss + g_target_loss

        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), gen_imgs

    def _train_discriminator(self, real_imgs, gen_imgs, valid, fake):
        """
        Trains the discriminator for one step on real and generated images.
        """
        self.d_optimizer.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_pred = self.discriminator(real_imgs)
        d_real_loss = self.adversarial_loss(real_pred, valid)
        
        # We detach gen_imgs so we don't calculate gradients for the generator again
        fake_pred = self.discriminator(gen_imgs.detach())
        d_fake_loss = self.adversarial_loss(fake_pred, fake)
        
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()

    def train_step(self, real_imgs):
        """
        Performs a single training step (one batch).
        """
        batch_size = real_imgs.size(0)
        
        # Ground truth labels (1 for real, 0 for fake)
        # Using soft labels (0.9 instead of 1.0, 0.1 instead of 0.0) for real images often helps GAN stability
        valid = torch.full((batch_size, 1), 0.9, dtype=torch.float, device=self.device)
        fake = torch.full((batch_size, 1), 0.1, dtype=torch.float, device=self.device)

        d_loss_total = 0.0
        for _ in range(self.discriminator_steps):
            z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
            with torch.no_grad():
                gen_imgs = self.generator(z)
            d_loss_total += self._train_discriminator(real_imgs, gen_imgs, valid, fake)

        g_loss_total = 0.0
        for _ in range(self.generator_steps):
            g_loss, _ = self._train_generator(batch_size)
            g_loss_total += g_loss

        g_loss = g_loss_total / self.generator_steps
        d_loss = d_loss_total / self.discriminator_steps
        
        return g_loss, d_loss

    def train(self, dataloader, epochs):
        """
        Main training loop over multiple epochs.
        """
        print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            # Use tqdm for a progress bar
            loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            
            for i, (imgs, _) in enumerate(loop):
                imgs = imgs.to(self.device)
                
                g_loss, d_loss = self.train_step(imgs)
                
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                
                # Update progress bar
                loop.set_postfix({"D loss": f"{d_loss:.4f}", "G loss": f"{g_loss:.4f}"})
                
            avg_g_loss = sum(g_losses) / len(g_losses)
            avg_d_loss = sum(d_losses) / len(d_losses)
            
            print(f"Epoch [{epoch+1}/{epochs}] | D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f}")
            
        print("Training completed.")
