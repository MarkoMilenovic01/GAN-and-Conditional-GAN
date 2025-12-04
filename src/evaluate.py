import torch
from torchvision.utils import save_image
from model import Generator
import os


os.makedirs("gan_evaluation", exist_ok=True)


def load_generator(checkpoint_path, device):
    G = Generator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    return G


def generate_samples(G, device, filename="generated.png"):
    """Generate 25 random samples and save them."""
    noise = torch.randn(25, 100).to(device)
    with torch.no_grad():
        samples = G(noise).cpu()
    save_image(samples, f"gan_evaluation/{filename}", nrow=5, normalize=True)
    print(f"Saved: gan_evaluation/{filename}")


def generator_evolution(device):
    """Generate evolution images using the same noise vector."""
    noise = torch.randn(25, 100).to(device)

    epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for epoch in epochs:
        ckpt = f"gan_checkpoints/G_epoch_{epoch}.pth"

        if not os.path.exists(ckpt):
            print(f"Checkpoint {ckpt} not found, skipping.")
            continue

        G = load_generator(ckpt, device)
        with torch.no_grad():
            samples = G(noise).cpu()

        save_image(samples, f"gan_evaluation/evolution_epoch_{epoch}.png",
                   nrow=5, normalize=True)

        print(f"Saved evolution image for epoch {epoch}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluating final generator...")
    G = load_generator("gan_checkpoints/G_epoch_100.pth", device)
    generate_samples(G, device, "final_samples.png")

    print("Generating generator evolution images...")
    generator_evolution(device)

    print("Evaluation complete.")
