"""
Visual similarity candidate retrieval module using pre-trained MobileNetV3 with GPU acceleration.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class ImageRecall:
    """Visual similarity candidate retrieval based on deep convolutional image embeddings."""

    def __init__(self, train_data, user_purchased_hist, images_dir,
                 embedding_path='data/processed/image_embeddings.npy'):
        self.train_data = train_data
        self.user_purchased_hist = user_purchased_hist
        self.images_dir = Path(images_dir)
        self.embedding_path = embedding_path
        self.embeddings = None
        self.article_ids = None
        self.article_to_idx = {}

        Path(embedding_path).parent.mkdir(parents=True, exist_ok=True)

    def extract_embeddings(self, batch_size=128):
        """Extracts visual embeddings for catalog images using pre-trained MobileNetV3."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device allocated for visual feature extraction: {device}")

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_files = list(self.images_dir.glob('**/*.jpg'))
        print(f"Catalog images found: {len(image_files):,}")

        embeddings_list = []
        article_ids_list = []

        batch_images = []
        batch_ids = []

        for img_path in tqdm(image_files, desc="Extracting image embeddings", leave=False):
            try:
                article_id = int(img_path.stem)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                batch_images.append(img_tensor)
                batch_ids.append(article_id)

                if len(batch_images) >= batch_size:
                    batch_tensor = torch.cat(batch_images, dim=0).to(device)
                    with torch.no_grad():
                        emb = model(batch_tensor).cpu().numpy()
                    embeddings_list.append(emb)
                    article_ids_list.extend(batch_ids)
                    batch_images = []
                    batch_ids = []
            except Exception:
                continue

        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            with torch.no_grad():
                emb = model(batch_tensor).cpu().numpy()
            embeddings_list.append(emb)
            article_ids_list.extend(batch_ids)

        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list)
            self.article_ids = np.array(article_ids_list)
            self.article_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}

            np.save(self.embedding_path, self.embeddings)
            pd.Series(self.article_ids).to_csv(
                str(self.embedding_path).replace('.npy', '_ids.csv'),
                index=False,
                header=['article_id']
            )
            print(f"Visual embeddings saved successfully: {self.embeddings.shape}")
        else:
            print("No image embeddings could be extracted.")

    def load_embeddings(self):
        """Loads precomputed visual embeddings from storage."""
        if os.path.exists(self.embedding_path):
            self.embeddings = np.load(self.embedding_path)
            ids_path = str(self.embedding_path).replace('.npy', '_ids.csv')
            if os.path.exists(ids_path):
                self.article_ids = pd.read_csv(ids_path)['article_id'].values
            self.article_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
            print(f"Visual embeddings loaded: {self.embeddings.shape}")
            return True
        return False

    def normalize_embeddings(self):
        """Performs L2 normalization across image embedding vectors."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms
        print("Visual embeddings normalized.")

    def recall(self, uid, top_k=100):
        """Retrieves candidates using cosine similarity to user visual profile."""
        if self.embeddings is None:
            return []

        purchased = self.user_purchased_hist.get(uid, set())
        purchased_list = list(purchased)

        if not purchased_list:
            return []

        recent_items = purchased_list[-10:]

        user_vector = None
        count = 0
        for item in recent_items:
            if item in self.article_to_idx:
                emb = self.embeddings[self.article_to_idx[item]]
                if user_vector is None:
                    user_vector = emb.copy()
                else:
                    user_vector += emb
                count += 1

        if user_vector is None or count == 0:
            return []

        user_vector = user_vector / count

        batch_size = 8192
        scores_list = []
        for start in range(0, len(self.embeddings), batch_size):
            end = min(start + batch_size, len(self.embeddings))
            batch_scores = self.embeddings[start:end].dot(user_vector)
            scores_list.append(batch_scores)

        all_scores = np.concatenate(scores_list)

        top_indices = np.argsort(all_scores)[::-1]
        results = []
        for idx in top_indices:
            item_id = self.article_ids[idx]
            if item_id not in purchased:
                results.append(item_id)
                if len(results) >= top_k:
                    break

        return results