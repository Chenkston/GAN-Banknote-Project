"""
Script to train Architecture B: DCGAN.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

from src.data.dataset import NoteShieldDataset
from src.models.dcgan import DCGANGenerator, DCGANDiscriminator
from src.training.trainer import GANTrainer
from src.data.preprocess import denormalize_segments

from concurrent.futures import ThreadPoolExecutor
import threading

def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/raw/modified_dataset"
    batch_size = 64
    epochs = 300
    latent_dim = 100
    img_size = (224, 224) # Using a smaller size initially to ensure stability and speed
    channels = 3
    img_shape = (6, channels, img_size[0], img_size[1])
    
    # Optimizers
    lr_G = 0.0002
    lr_D = 0.00005
    b1 = 0.5
    b2 = 0.999

    print(f"Using device: {device}")
    os.makedirs("outputs/dcgan/samples", exist_ok=True)
    os.makedirs("checkpoints/dcgan", exist_ok=True)

    # 2. Data Preparation
    # For GANs, we MUST normalize images to [-1, 1] because the Generator uses Tanh
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Maps [0, 1] -> [-1, 1]
    ])

    dataset = NoteShieldDataset(data_dir=data_dir, transform=transform, img_size=img_size)
    
    if len(dataset) == 0:
        print(f"Error: No data found in {data_dir}. Ensure 'real_notes' and 'fake_notes' folders exist.")
        return

    # Pre-load entire dataset onto GPU because I am bottlenecking on the CPU to GPU transfer
    print("Pre-loading dataset onto GPU...")
    all_imgs = [None] * len(dataset)
    all_labels = [None] * len(dataset)

    def load_sample(i):
        imgs, label = dataset[i]
        all_imgs[i] = imgs
        all_labels[i] = label

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(load_sample, range(len(dataset))))

    print("Stacking and moving to GPU...")
    all_imgs_tensor = torch.stack(all_imgs).to(device)
    all_labels_tensor = torch.tensor(all_labels).to(device)
    print(f"Dataset loaded onto GPU. Shape: {all_imgs_tensor.shape}")

    # Create a simple GPU-based dataloader using TensorDataset
    gpu_dataset = TensorDataset(all_imgs_tensor, all_labels_tensor)
    dataloader = DataLoader(gpu_dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Models
    generator = DCGANGenerator(latent_dim=latent_dim, img_shape=img_shape).to(device)
    discriminator = DCGANDiscriminator(img_shape=img_shape).to(device)

    # 4. Initialize Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(b1, b2))

    # 5. Initialize Trainer
    # Note: We do NOT use the TargetModel/Custom Loss for the baseline DCGAN
    # we want it to be a pure, standard GAN baseline.
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=optimizer_G,
        d_optimizer=optimizer_D,
        device=device
    )

    # 6. Custom Training logic to save samples
    print(f"Starting DCGAN training for {epochs} epochs...")
    
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
            generator.eval() 
            with torch.no_grad():
                z = torch.randn(1, latent_dim, device=device)
                gen_sample = generator(z).squeeze(0) # Get the 6 segments for one note (6, C, H, W)
                
                # Denormalize for viewing
                gen_sample = denormalize_segments(gen_sample)
                
                # Save as a grid of 6 images
                save_image(gen_sample, f"outputs/dcgan/samples/epoch_{epoch+1}.png", nrow=3)
            generator.train()

        # Save checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f"checkpoints/dcgan/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/dcgan/discriminator_epoch_{epoch+1}.pth")

    print("DCGAN training completed.")

if __name__ == "__main__":
    main()
