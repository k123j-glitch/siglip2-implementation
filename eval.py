import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_siglip2_base
from dataloader import FlickrDataset


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FlickrDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    model = get_siglip2_base().to(device)

    try:
        checkpoint = torch.load("siglip2_checkpoint.pth", map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print("Warning: checkpoint not found. Evaluating with random weights.")

    model.eval()

    all_image_embeds = []
    all_text_embeds = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for images, input_ids in tqdm(loader):
            images = images.to(device)
            input_ids = input_ids.to(device)

            v_raw = model.vision(images)
            v_embed = F.normalize(model.v_proj(v_raw), dim=-1)
            t_raw = model.text(input_ids)
            t_embed = F.normalize(model.t_proj(t_raw), dim=-1)

            all_image_embeds.append(v_embed)
            all_text_embeds.append(t_embed)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    similarity = all_image_embeds @ all_text_embeds.t()
    ground_truth = torch.arange(len(all_image_embeds), device=device)

    # Image-to-Text (I2T) - For each image, find the best text
    i2t_prd = similarity.argmax(dim=1)
    i2t_acc = (i2t_prd == ground_truth).float().mean()

    # Text-to-Image (T2I) - For each text, find the best image
    t2i_prd = similarity.argmax(dim=0)
    t2i_acc = (t2i_prd == ground_truth).float().mean()

    print(f"\n--- Evaluation Results ---")
    print(f"Image-to-Text (I2T) Rank@1: {i2t_acc.item() * 100:.2f}%")
    print(f"Text-to-Image (T2I) Rank@1: {t2i_acc.item() * 100:.2f}%")


if __name__ == "__main__":
    evaluate()