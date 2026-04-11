"""
Script to train the Architecture A: Multi-Layer Perceptron (MLP) GAN (Baseline).
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

from src.data.dataset import NoteShieldDataset
from src.models.mlp_gan import MLPGenerator, MLPDiscriminator
from src.training.trainer import GANTrainer
from src.data.preprocess import denormalize_segments

def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"
    batch_size = 64
    epochs = 200
    latent_dim = 100
    img_size = (64, 64) # Using a smaller size for the MLP baseline to ensure stability and speed
    channels = 3
    img_shape = (6, channels, img_size[0], img_size[1])
    
    # Optimizers
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999

    print(f"Using device: {device}")
    os.makedirs("outputs/mlp/samples", exist_ok=True)
    os.makedirs("checkpoints/mlp", exist_ok=True)

    # 2. Data Preparation
    # For GANs, we MUST normalize images to [-1, 1] because the Generator uses Tanh
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Maps [0, 1] -> [-1, 1]
    ])

    dataset = NoteShieldDataset(data_dir=data_dir, transform=transform)
    
    if len(dataset) == 0:
        print(f"Error: No data found in {data_dir}. Ensure 'real_notes' and 'fake_notes' folders exist.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 3. Initialize Models
    generator = MLPGenerator(latent_dim=latent_dim, img_shape=img_shape).to(device)
    discriminator = MLPDiscriminator(img_shape=img_shape).to(device)

    # 4. Initialize Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # 5. Initialize Trainer
    # Note: We do NOT use the TargetModel/Custom Loss for the baseline MLP GAN
    # we want it to be a pure, standard GAN baseline.
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=optimizer_G,
        d_optimizer=optimizer_D,
        device=device
    )

    # 6. Custom Training logic to save samples
    print(f"Starting MLP GAN training for {epochs} epochs...")
    
    for epoch in range(epochs):
        g_loss_total = 0
        d_loss_total = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # Use the trainer's step
            g_loss, d_loss = trainer.train_step(imgs)
            
            g_loss_total += g_loss
            d_loss_total += d_loss

        # Calculate averages
        avg_g = g_loss_total / len(dataloader)
        avg_d = d_loss_total / len(dataloader)
        
        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {avg_d:.4f}] [G loss: {avg_g:.4f}]")

        # Periodically save image samples
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z = torch.randn(1, latent_dim, device=device)
                gen_sample = generator(z).squeeze(0) # Get the 6 segments for one note (6, C, H, W)
                
                # Denormalize for viewing
                gen_sample = denormalize_segments(gen_sample)
                
                # Save as a grid of 6 images
                save_image(gen_sample, f"outputs/mlp/samples/epoch_{epoch+1}.png", nrow=3)

        # Save checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f"checkpoints/mlp/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/mlp/discriminator_epoch_{epoch+1}.pth")

    print("MLP GAN training completed.")

if __name__ == "__main__":
    main()
