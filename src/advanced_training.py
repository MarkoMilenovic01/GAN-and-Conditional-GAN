import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from advanced_model import cGenerator, cDiscriminator, one_hot
from dataset import get_mnist_dataloaders

import os


os.makedirs("cgan_results", exist_ok=True)
os.makedirs("cgan_checkpoints", exist_ok=True)


def train_cgan():

    batch_size = 128
    latent_dim = 100
    lr = 0.0002
    betas = (0.5, 0.999)
    num_epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = get_mnist_dataloaders(batch_size=batch_size)

    G = cGenerator(noise_dim=latent_dim).to(device)
    D = cDiscriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)

    # Fixed noise and labels for visualization
    fixed_noise = torch.randn(25, latent_dim).to(device)
    fixed_labels = torch.randint(0, 10, (25,), device=device)
    fixed_labels_oh = one_hot(fixed_labels, 10, device)

    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        for real_imgs, labels in train_loader:

            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            N = real_imgs.size(0)

            labels_oh = one_hot(labels, 10, device)  # for Generator ONLY

            # ---------------------
            #    Train D
            # ---------------------
            D.zero_grad()

            # real
            real_targets = torch.ones(N, 1).to(device)
            pred_real = D(real_imgs, labels)     # integer labels
            loss_real = criterion(pred_real, real_targets)
            loss_real.backward()

            # fake
            noise = torch.randn(N, latent_dim).to(device)
            fake_imgs = G(noise, labels_oh)
            fake_targets = torch.zeros(N, 1).to(device)
            pred_fake = D(fake_imgs.detach(), labels)   # integer labels
            loss_fake = criterion(pred_fake, fake_targets)
            loss_fake.backward()

            optimizer_D.step()

            D_loss = loss_real + loss_fake

            # ---------------------
            #    Train G
            # ---------------------
            G.zero_grad()

            noise = torch.randn(N, latent_dim).to(device)
            fake_imgs = G(noise, labels_oh)

            target_labels = torch.ones(N, 1).to(device)
            pred = D(fake_imgs, labels)   # integer labels
            loss_G = criterion(pred, target_labels)

            loss_G.backward()
            optimizer_G.step()

            G_losses.append(loss_G.item())
            D_losses.append(D_loss.item())

        # Save sample grid each epoch
        with torch.no_grad():
            samples = G(fixed_noise, fixed_labels_oh).cpu()
            save_image(samples, f"cgan_results/epoch_{epoch}.png", nrow=5, normalize=True)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss_D: {D_loss.item():.4f}  Loss_G: {loss_G.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f"cgan_checkpoints/G_epoch_{epoch+1}.pth")
            torch.save(D.state_dict(), f"cgan_checkpoints/D_epoch_{epoch+1}.pth")

    # Final samples
    with torch.no_grad():
        final_noise = torch.randn(25, latent_dim).to(device)
        final_labels = torch.randint(0, 10, (25,), device=device)
        final_labels_oh = one_hot(final_labels, 10, device)
        final_samples = G(final_noise, final_labels_oh).cpu()
        save_image(final_samples, "cgan_results/final.png", nrow=5, normalize=True)

    print("Training complete.")

    return G_losses, D_losses


# ============================
#   LOSS CURVES (like GAN)
# ============================

import matplotlib.pyplot as plt

if __name__ == "__main__":
    G_losses, D_losses = train_cgan()

    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("cGAN Training Loss Curves")
    plt.legend()
    plt.savefig("cgan_results/loss_curves.png")
    plt.show()
