import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_siglip_1 import SigLIP1
from dataloader import FlickrDataset


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    dataset = FlickrDataset()

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0  # safer on Windows
    )

    model = SigLIP1(vocab_size=dataset.vocab_size).to(device)

    try:
        checkpoint = torch.load("siglip1_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print("Warning: checkpoint not found. Evaluating random weights.")

    model.eval()

    all_image_embeds = []
    all_text_embeds = []

    print("Extracting embeddings...")

    with torch.no_grad():
        for images, input_ids in tqdm(loader):

            images = images.to(device)
            input_ids = input_ids.to(device)

            # Use model forward pieces properly
            v = model.vision(images)
            t = model.text(input_ids)

            v = F.normalize(model.v_proj(v), dim=-1)
            t = F.normalize(model.t_proj(t), dim=-1)

            all_image_embeds.append(v)
            all_text_embeds.append(t)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    similarity = all_image_embeds @ all_text_embeds.t()

    ground_truth = torch.arange(len(all_image_embeds), device=device)

    # Image → Text
    i2t_pred = similarity.argmax(dim=1)
    i2t_acc = (i2t_pred == ground_truth).float().mean()

    # Text → Image
    t2i_pred = similarity.argmax(dim=0)
    t2i_acc = (t2i_pred == ground_truth).float().mean()

    print("\n--- Evaluation Results ---")
    print(f"Image-to-Text (I2T) Rank@1: {i2t_acc.item() * 100:.2f}%")
    print(f"Text-to-Image (T2I) Rank@1: {t2i_acc.item() * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
