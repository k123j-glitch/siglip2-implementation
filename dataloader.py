# dataloader.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FlickrDataset(Dataset):
    def __init__(self, root_dir="Flickr8k_Dataset", max_len=32):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "Images")
        self.max_len = max_len

        self.samples = []

        with open(os.path.join(root_dir, "captions.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines[1:]:
            try:
                image_name, caption = line.strip().split(",", 1)
                self.samples.append((image_name, caption))
            except ValueError:
                continue

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        self.build_vocab()

    # ------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------

    def build_vocab(self):
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<cls>": 2,
            "<eos>": 3,
        }

        idx = 4

        for _, caption in self.samples:
            for word in caption.lower().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1

        self.vocab = vocab
        self.vocab_size = len(vocab)

    # ------------------------------------------------------------

    def tokenize(self, caption):

        tokens = [self.vocab["<cls>"]]

        for word in caption.lower().split():
            tokens.append(self.vocab.get(word, self.vocab["<unk>"]))

        tokens.append(self.vocab["<eos>"])

        if len(tokens) < self.max_len:
            tokens += [self.vocab["<pad>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.tensor(tokens, dtype=torch.long)

    # ------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        input_ids = self.tokenize(caption)

        # IMPORTANT: return EXACTLY TWO ITEMS
        return image, input_ids
