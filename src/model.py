import torch
import torch.nn as nn


# ============================
#       DISCRIMINATOR
# ============================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Input: [N, 1, 28, 28]
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),   # [N, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # [N, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),                                           # [N, 3136]
            nn.Linear(3136, 1),                                     # [N, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ============================
#         GENERATOR
# ============================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(100, 6272),  # 100 → 6272
            nn.ReLU(True)
        )

        # After reshape: [N, 128, 7, 7]
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # [N, 128, 28, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3),  # [N, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)               # [N, 6272]
        x = x.view(-1, 128, 7, 7)    # reshape to [N, 128, 7, 7]
        return self.deconv(x)


# ============================
#         SELF-TEST
# ============================

if __name__ == "__main__":
    D = Discriminator()
    G = Generator()

    z = torch.randn(16, 100)
    fake = G(z)
    out = D(fake)

    print("Fake image shape: ", fake.shape)   # Expect [16, 1, 28, 28]
    print("Discriminator output shape:", out.shape)  # Expect [16, 1]
