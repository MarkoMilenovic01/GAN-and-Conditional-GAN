import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from model import Generator, Discriminator
from dataset import get_mnist_dataloaders

import os

import matplotlib.pyplot as plt



os.makedirs("gan_results", exist_ok=True)
os.makedirs("gan_checkpoints", exist_ok=True)


def train_gan():

    # ============= HYPERPARAMETERS =============
    batch_size = 128
    latent_dim = 100
    lr = 0.0002
    betas = (0.5, 0.999)
    num_epochs = 100

    # ============= DEVICE =============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============= DATA =============
    train_loader, _ = get_mnist_dataloaders(batch_size=batch_size)

    # ============= MODELS =============
    G = Generator().to(device)
    D = Discriminator().to(device)

    # ============= LOSS + OPTIMIZERS =============
    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)

    # ============= FIXED NOISE =============
    fixed_noise = torch.randn(25, latent_dim).to(device)

    # ============= LOSS CURVES STORAGE =============
    G_losses = []
    D_losses = []

    # ============= TRAINING LOOP =============
    for epoch in range(num_epochs):
        for real_imgs, _ in train_loader:

            real_imgs = real_imgs.to(device)
            N = real_imgs.size(0)

            # =========================================
            #               TRAIN D
            # =========================================
            D.zero_grad()

            # Real
            real_labels = torch.ones(N, 1).to(device)
            pred_real = D(real_imgs)
            loss_real = criterion(pred_real, real_labels)
            loss_real.backward()

            # Fake
            noise = torch.randn(N, latent_dim).to(device)
            fake_imgs = G(noise)
            fake_labels = torch.zeros(N, 1).to(device)
            pred_fake = D(fake_imgs.detach())
            loss_fake = criterion(pred_fake, fake_labels)
            loss_fake.backward()

            optimizer_D.step()

            D_loss = loss_real + loss_fake

            # =========================================
            #               TRAIN G
            # =========================================
            G.zero_grad()
            noise = torch.randn(N, latent_dim).to(device)
            fake_imgs = G(noise)

            target_labels = torch.ones(N, 1).to(device)
            pred = D(fake_imgs)
            loss_G = criterion(pred, target_labels)

            loss_G.backward()
            optimizer_G.step()

            # Track losses
            G_losses.append(loss_G.item())
            D_losses.append(D_loss.item())

        # Save sample grid each epoch
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            save_image(samples, f"gan_results/epoch_{epoch}.png", nrow=5, normalize=True)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss_D: {D_loss.item():.4f}  Loss_G: {loss_G.item():.4f}")

        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f"gan_checkpoints/G_epoch_{epoch+1}.pth")
            torch.save(D.state_dict(), f"gan_checkpoints/D_epoch_{epoch+1}.pth")

    # Final samples
    with torch.no_grad():
        final_noise = torch.randn(25, latent_dim).to(device)
        final_samples = G(final_noise).cpu()
        save_image(final_samples, "gan_results/final.png", nrow=5, normalize=True)

    print("Training complete.")

    return G_losses, D_losses




if __name__ == "__main__":
    G_losses, D_losses = train_gan()



    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.savefig("gan_results/loss_curves.png")
    plt.show()
