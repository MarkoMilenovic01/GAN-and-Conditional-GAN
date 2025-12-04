import torch
from torchvision.utils import save_image
from advanced_model import cGenerator, one_hot
import os


def evaluate_cgan(checkpoint_path, digit, num_samples=25, out_path="cgan_results/eval.png"):
    """
    Load a trained cGAN generator and generate images for a single digit.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- LOAD GENERATOR --------
    G = cGenerator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    # -------- PREP INPUT --------
    noise = torch.randn(num_samples, 100).to(device)

    # Digit labels
    labels = torch.full((num_samples,), digit, dtype=torch.long, device=device)
    labels_oh = one_hot(labels, 10, device)   # For generator ONLY

    # -------- GENERATE --------
    with torch.no_grad():
        samples = G(noise, labels_oh).cpu()

    # -------- SAVE --------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(samples, out_path, nrow=int(num_samples**0.5), normalize=True)

    print(f"Generated {num_samples} samples of digit {digit}")
    print(f"Saved to: {out_path}")


# ==========================
#         SIMPLE MAIN
# ==========================
if __name__ == "__main__":

    # CHANGE THESE:
    checkpoint = "cgan_checkpoints/G_epoch_100.pth"
    digit = 5
    num_samples = 25

    evaluate_cgan(
        checkpoint_path=checkpoint,
        digit=digit,
        num_samples=num_samples,
        out_path="cgan_results/eval.png"
    )
