import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime

from model_siglip_1 import siglip1_loss, SigLIP1
from dataloader import FlickrDataset


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 10
    lr = 1e-4
    weight_decay = 0.01

    print("Loading dataset...")
    dataset = FlickrDataset()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    vocab_size = dataset.vocab_size
    model = SigLIP1(vocab_size=vocab_size).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98)
    )

    total_steps = epochs * len(loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )

    scaler = torch.amp.GradScaler()

    # -----------------------
    # TensorBoard Setup
    # -----------------------
    log_dir = f"runs/siglip1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    os.makedirs("checkpoints", exist_ok=True)

    print(f"Starting training on {device}...")
    print(f"Total steps: {total_steps}")

    global_step = 0

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, input_ids in pbar:

            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits = model(images, input_ids)
                loss = siglip1_loss(logits)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            # -----------------------
            # TensorBoard Logging
            # -----------------------
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)

            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        avg_loss = epoch_loss / len(loader)

        print(f"\nFinished Epoch {epoch+1}. Average Loss: {avg_loss:.4f}")

        # Log epoch average
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        # -----------------------
        # Save checkpoint
        # -----------------------
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
        }

        torch.save(checkpoint, f"checkpoints/siglip1_epoch_{epoch+1}.pth")
        torch.save(checkpoint, "checkpoints/siglip1_latest.pth")

        print(f"Checkpoint saved for epoch {epoch+1}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
