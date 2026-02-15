import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TurbulenceDataset
from model import TurbulenceUNet
from loss import PhysicsLoss, DivergenceLoss

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
SAVE_INTERVAL = 10

# Loss Weights
W_CONTENT = 1.0
W_PHYSICS = 0.1
W_DIV = 0.5

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/turbulence_data.npy")
TARGET_PATH = os.path.join(BASE_DIR, "../data/processed/target_spectrum.npy")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "../experiments/checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    train_dataset = TurbulenceDataset(DATA_PATH, mode='train')
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TurbulenceUNet(upscale_factor=16).to(device)
    loss_phys = PhysicsLoss(TARGET_PATH, device=device, size=1024)
    loss_div = DivergenceLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(train_dataset)} images.")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()

        epoch_metrics = {'total': 0.0, 'content': 0.0, 'phys': 0.0, 'div': 0.0}

        for batch_idx, coarse_imgs in enumerate(train_loader):
            coarse_imgs = coarse_imgs.to(device)
            optimizer.zero_grad()

            coarse_down = F.adaptive_avg_pool2d(coarse_imgs, (32, 32))
            pred_512 = model(coarse_down)
            pred_verify = F.adaptive_avg_pool2d(pred_512, (64, 64))

            loss_content = F.mse_loss(pred_verify, coarse_imgs)

            high_res_output = model(coarse_imgs)
            loss_physics = loss_phys(high_res_output)
            loss_divergence = loss_div(high_res_output)

            total_loss = (W_CONTENT * loss_content) + \
                         (W_PHYSICS * loss_physics) + \
                         (W_DIV * loss_divergence)

            total_loss.backward()
            optimizer.step()

            epoch_metrics['total'] += total_loss.item()
            epoch_metrics['content'] += loss_content.item()
            epoch_metrics['phys'] += loss_physics.item()
            epoch_metrics['div'] += loss_divergence.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"[Epoch {epoch}][Batch {batch_idx+1}] "
                      f"Tot: {total_loss.item():.4f} | "
                      f"Con: {loss_content.item():.4f} | "
                      f"Spec: {loss_physics.item():.4f} | "
                      f"Div: {loss_divergence.item():.4f}")

        avg_loss = epoch_metrics['total'] / len(train_loader)
        duration = time.time() - start_time
        print(
            f"=== Epoch {epoch} Finished ({duration:.1f}s) | Avg Loss: {avg_loss:.4f} ===")

        if epoch % SAVE_INTERVAL == 0:
            save_path = os.path.join(
                CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    train()
