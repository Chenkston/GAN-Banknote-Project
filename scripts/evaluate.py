"""Script to evaluate a trained GAN against the target model."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from src.data.dataset import NoteShieldDataset
from src.models.dcgan import DCGANGenerator
from src.models.mlp_gan import MLPGenerator
from src.models.target_model import TargetCNN
from src.evaluation.metrics import calculate_fooling_rate, calculate_fid
from src.data.preprocess import denormalize_segments

def make_real_segments_tensor(dataset):
    """
    Gets the images from the dataset, and turns them into a tensor.

    Args:
        dataset (NoteShieldDataset): The dataset with the images

    Returns:
        Tensor: Contains the tensor version of the dataset.
    """
    all_segments = []
    for i in range(len(dataset)):
        imgs, _ = dataset[i]  # (6, C, H, W)
        all_segments.append(imgs)
    flat = torch.stack(all_segments)       # (N, 6, C, H, W)
    N, S, C, H, W = flat.shape
    return flat.view(N * S, C, H, W)      # (N*6, C, H, W), stays on CPU

def get_or_generate_fake_notes(generator, latent_dim, num_samples, batch_size, device, cache_path):
    """
    Load fake notes from cache if available, otherwise generate and save them. This was added to save
    computation time, generating a fresh set of banknotes every time. To generate a new dataset just
    delete the folder at cache_path.

    Args:
        generator (nn.Module): Model that takes in a latent vector, and turns it into a 6-segment bank note image.
        latent_dim (Int): The dimensionality of the latent vector.
        num_samples (Int): The number of bank notes to generate.
        batch_size (Int): The number of bank notes to generate in a single call to the model.
        device (torch.device): The device on which to run the model.
        cache_path (Path): The path to where the fake bank note dataset should be saved / loaded from.

    Returns:
        Tensor: A tensor version of the fake bank note dataset.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached fake notes from {cache_path}...")
        return torch.load(cache_path)

    print(f"Generating {num_samples} fake notes...")
    generator.eval()
    all_notes = []
    total = 0

    with torch.no_grad():
        while total < num_samples:
            current_batch = min(batch_size, num_samples - total)
            z = torch.randn(current_batch, latent_dim, device=device)
            fake_notes = generator(z).cpu()
            all_notes.append(fake_notes)
            total += current_batch

    fake_notes_tensor = torch.cat(all_notes, dim=0)  # (N, 6, C, H, W)
    torch.save(fake_notes_tensor, cache_path)
    print(f"Saved fake notes to {cache_path}")
    return fake_notes_tensor


def evaluate_generator(name, generator, latent_dim, target_model,
                        real_segs, num_samples, batch_size, device, output_dir):
    """
    Evaluates the generator against the target model, also gets the FID score.

    Args:
        name (String): The name of the generator model being evaluated
        generator (nn.Module): Model that takes in a latent vector, and turns it into a 6-segment bank note image.
        latent_dim (Int): The dimensionality of the latent vector.
        target_model (nn.Module): The densenet121 model trained on the real bank note dataset.
        real_segs (Tensor): The tensor version of the real banknote dataset
        num_samples (Int): The number of bank notes to generate.
        batch_size (Int): The number of bank notes to generate in a single call to the model.
        device (torch.device): The device on which to run the model.
        output_dir (Path): The path to where sample generated notes should be stored.

    Returns:
        Dict: A dict mapping the model name to its fooling rate and FID score.
    """
    print(f"\n{'='*50}\nEvaluating: {name}\n{'='*50}")
    os.makedirs(f"{output_dir}/{name}/samples", exist_ok=True)

    # Get already generated fake notes if they exist, make them if they don't
    cache_path = os.path.join(output_dir, name, "fake_notes.pt")
    fake_notes = get_or_generate_fake_notes(
        generator, latent_dim, num_samples, batch_size, device, cache_path
    )

    # Fund the fooling rate, how often the generator can fool the target_model
    target_model.eval()
    fooling_rate = calculate_fooling_rate(fake_notes=fake_notes, target_model=target_model, batch_size=batch_size)
    print(f"[{name}] Fooling Rate: {fooling_rate:.4f} ({fooling_rate*100:.2f}%)")

    # FID, a metric that approximates how real the images look relative to the real set.
    N, S, C, H, W = fake_notes.shape
    fake_segs = fake_notes.view(N * S, C, H, W)
    fid_score = calculate_fid(real_segs, fake_segs, batch_size)
    print(f"[{name}] FID Score:     {fid_score:.4f}")

    # Save 5 of the fake generated images
    with torch.no_grad():
        for i in range(min(5, len(fake_notes))):
            save_image(
                denormalize_segments(fake_notes[i]),
                f"{output_dir}/{name}/samples/sample_{i+1}.png",
                nrow=3
            )

    return {"name": name, "fooling_rate": fooling_rate, "fid_score": fid_score}


def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir   = "data/raw/modified_dataset"
    batch_size = 16
    latent_dim = 100
    img_size   = (224, 224)
    channels   = 3
    img_shape_dcgan = (6, channels, img_size[0], img_size[1])  # (6, 3, 224, 224)
    img_shape_mlp   = (6, channels, 64, 64)                    # (6, 3, 64, 64)
    num_samples = 200  # notes to generate; total segments evaluated = num_samples * 6

    checkpoint_dir = "checkpoints"
    output_dir     = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    dcgan_checkpoint  = os.path.join(checkpoint_dir, "dcgan",   "generator_epoch_300.pth")
    mlp_checkpoint    = os.path.join(checkpoint_dir, "mlp", "generator_epoch_300.pth")
    target_checkpoint = os.path.join(checkpoint_dir, "target_model_best.pth")

    # 2. Data
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = NoteShieldDataset(data_dir=data_dir, transform=transform, img_size=img_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No data found in {data_dir}.")
    print(f"Found {len(dataset)} samples.")

    # Tensor representing real image dataset
    real_segs = make_real_segments_tensor(dataset)  # used for both generators

    # 3. Load the Target Model
    print("\nLoading target model...")
    target_model = TargetCNN().to(device)
    target_model.load_state_dict(torch.load(target_checkpoint, map_location=device))
    target_model.eval()
    print("Target model loaded.")

    # 4. Evaluate both generators
    generators = [
        ("dcgan",   DCGANGenerator(latent_dim=latent_dim, img_shape=img_shape_dcgan)),
        ("mlp_gan", MLPGenerator(latent_dim=latent_dim, img_shape=img_shape_mlp)),
    ]

    results = []
    for name, generator in generators:
        ckpt = dcgan_checkpoint if name == "dcgan" else mlp_checkpoint
        print(f"\nLoading {name} checkpoint from {ckpt}...")
        generator.load_state_dict(torch.load(ckpt, map_location=device))
        generator = generator.to(device)

        result = evaluate_generator(
            name=name,
            generator=generator,
            latent_dim=latent_dim,
            target_model=target_model,
            real_segs=real_segs,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
        )
        results.append(result)

    # 5. Summarize the output
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"{'Generator':<12} {'Fooling Rate':>14} {'FID Score':>12}")
    print(f"{'-'*40}")
    for r in results:
        print(f"{r['name']:<12} {r['fooling_rate']:>13.2%} {r['fid_score']:>12.4f}")

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Generator,Fooling Rate,FID Score\n")
        for r in results:
            f.write(f"{r['name']},{r['fooling_rate']:.4f},{r['fid_score']:.4f}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()