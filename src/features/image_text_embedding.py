"""
Multimodal image and text embedding extraction module.

Key Capabilities:
1. Image embeddings: Extracts visual representations via pre-trained ResNet-50.
2. Text embeddings: Computes semantic item representations via TF-IDF + TruncatedSVD (fully offline).
3. Dimensionality reduction: Compresses representations to 32 dimensions with persistent caching.

Usage:
    extractor = ImageTextEmbeddingExtractor(articles, images_dir)
    image_emb, text_emb = extractor.extract_all()
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ImageTextEmbeddingExtractor:
    """Multimodal image and text embedding extractor."""

    def __init__(self, articles, images_dir=None, cache_dir=None):
        self.articles = articles
        self.images_dir = Path(images_dir) if images_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_image_embeddings(self, batch_size=64, dim=32):
        """
        Extracts image embeddings using pre-trained ResNet-50 and reduces dimensions via PCA.
        """
        cache_file = self.cache_dir / f'image_emb_{dim}.pkl'
        if cache_file.exists():
            print(f"  Loading cached image embeddings: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        if self.images_dir is None or not self.images_dir.exists():
            print("  Skipping image embeddings (image directory not found).")
            return {}

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            from PIL import Image
            from sklearn.decomposition import PCA
        except ImportError as e:
            print(f"  Skipping image embeddings (missing dependencies: {e}).")
            return {}

        print("  Loading pre-trained ResNet-50 model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(device).eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        article_ids = self.articles['article_id'].tolist()
        raw_features = []
        valid_ids = []

        with torch.no_grad():
            batch_imgs = []
            batch_ids = []
            for aid in tqdm(article_ids, desc="  Extracting visual features", leave=False):
                aid_str = f"{aid:010d}"
                img_path = self.images_dir / aid_str[:3] / f"{aid_str}.jpg"
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_imgs.append(transform(img))
                    batch_ids.append(aid)
                except Exception:
                    continue

                if len(batch_imgs) >= batch_size:
                    batch_tensor = torch.stack(batch_imgs).to(device)
                    feats = model(batch_tensor).squeeze().cpu().numpy()
                    raw_features.append(feats)
                    valid_ids.extend(batch_ids)
                    batch_imgs = []
                    batch_ids = []

            if batch_imgs:
                batch_tensor = torch.stack(batch_imgs).to(device)
                feats = model(batch_tensor).squeeze().cpu().numpy()
                raw_features.append(feats)
                valid_ids.extend(batch_ids)

        if not raw_features:
            print("  No image representations extracted.")
            return {}

        all_feats = np.vstack(raw_features)
        print(f"  Raw feature dimension: {all_feats.shape}")

        pca = PCA(n_components=dim)
        reduced = pca.fit_transform(all_feats)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        reduced = reduced / norms

        embedding = {aid: reduced[i] for i, aid in enumerate(valid_ids)}
        print(f"  Image embeddings generated: {len(embedding):,} items, dimension={dim}")

        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
        return embedding

    def extract_text_embeddings(self, dim=32):
        """
        Extracts semantic text representations using TF-IDF and TruncatedSVD (fully offline).
        """
        cache_file = self.cache_dir / f'text_emb_tfidf_{dim}.pkl'
        if cache_file.exists():
            print(f"  Loading cached text embeddings: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("  Extracting text features via TF-IDF and SVD (offline)...")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            print("  Scikit-learn not available, skipping text embeddings.")
            return {}

        text_cols = ['prod_name', 'product_type_name', 'product_group_name',
                     'graphical_appearance_name', 'colour_group_name',
                     'perceived_colour_value_name', 'department_name',
                     'index_name', 'section_name', 'garment_group_name']
        text_cols = [c for c in text_cols if c in self.articles.columns]

        articles = self.articles.copy()
        articles['text'] = articles[text_cols].fillna('').agg(' '.join, axis=1)
        if 'detail_desc' in articles.columns:
            articles['text'] = articles['text'] + ' ' + articles['detail_desc'].fillna('')

        texts = articles['text'].tolist()
        article_ids = articles['article_id'].tolist()

        print(f"  Building TF-IDF representation ({len(texts):,} items)...")
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        print(f"  TF-IDF dimension: {tfidf_matrix.shape[1]}, reducing to {dim}...")
        svd = TruncatedSVD(n_components=dim, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)

        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        reduced = reduced / norms

        embedding = {aid: reduced[i] for i, aid in enumerate(article_ids)}
        print(f"  Text embeddings generated: {len(embedding):,} items, dimension={dim}")
        print(f"  SVD cumulative explained variance: {svd.explained_variance_ratio_.sum():.3f}")

        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
        return embedding

    def extract_all(self, dim=32):
        """
        Extracts both visual and textual embeddings.
        """
        print("=" * 60)
        print("Extracting Multimodal Image and Text Embeddings")
        print("=" * 60)
        image_emb = self.extract_image_embeddings(dim=dim)
        text_emb = self.extract_text_embeddings(dim=dim)
        return image_emb, text_emb