import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import get_siglip2_base, siglip2_loss
from dataloader import FlickrDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 10
    lr = 1e-5
    weight_decay = 0.01

    print("Loading dataset...")
    dataset = FlickrDataset()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    model = get_siglip2_base().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, input_ids in pbar:
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)

            logits = model(images, input_ids)
            loss = siglip2_loss(logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Finished Epoch {epoch + 1}. Average Loss: {avg_loss:.4f}")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, "siglip2_checkpoint.pth")
    print("Model saved to siglip2_checkpoint.pth")


if __name__ == "__main__":
    train()