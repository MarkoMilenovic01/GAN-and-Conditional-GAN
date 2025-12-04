import torch
import torch.nn as nn


# ==========================================================
#                 HELPER — ONE-HOT ENCODING
# ==========================================================
def one_hot(labels, num_classes=10, device="cpu"):
    """
    Convert integer labels to one-hot vectors.
    labels: Tensor of shape [N]
    output: Tensor of shape [N, 10]
    """
    return torch.eye(num_classes, device=device)[labels]


# ==========================================================
#                  cGAN GENERATOR
# ==========================================================
class cGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.input_dim = noise_dim + num_classes   # 100 + 10 = 110

        # Fully connected block
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 6272),  # → reshape to (128, 7, 7)
            nn.ReLU(True)
        )

        # Deconvolution block
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()  # Final MNIST image [0,1]
        )

    def forward(self, z, labels_onehot):
        # Concatenate noise + labels
        x = torch.cat([z, labels_onehot], dim=1)  # [N, 110]

        # FC → reshape
        x = self.fc(x)                  # [N, 6272]
        x = x.view(-1, 128, 7, 7)       # [N, 128, 7, 7]

        # Deconv upsampling
        x = self.deconv(x)
        return x  # [N, 1, 28, 28]


# ==========================================================
#                  cGAN DISCRIMINATOR
# ==========================================================
class cDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),  # [N, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # [N, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),                      # [N, 3136]
            nn.Linear(3136, 1),
            nn.Sigmoid()                       # Real or fake?
        )

    def forward(self, images, labels):
        """
        images: [N, 1, 28, 28]
        labels: [N]  (integer labels 0–9)
        """

        N = images.size(0)

        # =============================
        #  SIMPLE & CORRECT LABEL MAP
        # =============================
        # Label map = 28x28 filled with label number
        # Example: label=5 → 28×28 full of 5’s
        label_map = labels.view(N, 1, 1, 1).float()      # [N,1,1,1]
        label_map = label_map.expand(N, 1, 28, 28)       # [N,1,28,28]

        # Concatenate image + label channel
        x = torch.cat([images, label_map], dim=1)        # [N, 2, 28, 28]

        return self.model(x)


# ==========================================================
#                        MAIN TEST
# ==========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = torch.randn(16, 100).to(device)
    labels = torch.randint(0, 10, (16,), device=device)
    labels_oh = one_hot(labels, 10, device)

    G = cGenerator().to(device)
    fake_imgs = G(z, labels_oh)
    print("Generator output shape:", fake_imgs.shape)

    D = cDiscriminator().to(device)
    out = D(fake_imgs, labels)
    print("Discriminator output shape:", out.shape)

    print("advanced_model.py is working correctly.")

    print("Label:", labels[0].item())
    print("Label map:")
    print(labels[0].view(1,1,1,1).expand(1,1,5,5))  

