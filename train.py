"""Train BuildSense AI models from a labelled concrete crack image dataset.

Expected dataset layout (folder names can vary):
data/concrete_cracks/raw/
  Negative/   # non-cracked concrete images
  Positive/   # cracked concrete images

Run: python train.py --data-dir data/concrete_cracks/raw
The script writes public/model.json for static browser-side inference.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCA, LDA and neural-network concrete image models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/concrete_cracks/raw"),
                        help="Directory containing one subfolder per class.")
    parser.add_argument("--image-size", type=int, default=64, help="Square grayscale image size (default: 64).")
    parser.add_argument("--max-per-class", type=int, default=1200,
                        help="Maximum images loaded per class to keep PCA training manageable.")
    parser.add_argument("--components", type=int, default=60, help="PCA component count.")
    parser.add_argument("--output", type=Path, default=Path("public/model.json"), help="Output model JSON path.")
    return parser.parse_args()


def load_dataset(root: Path, image_size: int, max_per_class: int):
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\nDownload and extract the labelled concrete crack dataset, "
            "then pass the folder containing the class subfolders with --data-dir."
        )
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    class_dirs = [p for p in class_dirs if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.rglob("*") if f.is_file())]
    if len(class_dirs) != 2:
        raise ValueError(
            f"Expected exactly 2 class folders containing images inside {root}, found {len(class_dirs)}: "
            f"{[p.name for p in class_dirs]}. Point --data-dir at the folder directly containing the two class folders."
        )

    images, labels, paths = [], [], []
    for class_dir in class_dirs:
        candidates = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        random.Random(RANDOM_STATE).shuffle(candidates)
        accepted = 0
        for path in candidates:
            try:
                with Image.open(path) as img:
                    img = img.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS)
                    array = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
                images.append(array)
                labels.append(class_dir.name)
                paths.append(str(path))
                accepted += 1
            except (UnidentifiedImageError, OSError) as exc:
                print(f"Skipping unreadable image: {path} ({exc})")
            if accepted >= max_per_class:
                break
        print(f"Loaded {accepted:>5} images from class '{class_dir.name}'")
        if accepted == 0:
            raise ValueError(f"No readable images found in {class_dir}")
    return np.asarray(images, dtype=np.float32), np.asarray(labels), paths


def main():
    args = parse_args()
    if args.image_size < 16:
        raise ValueError("--image-size should be at least 16.")
    X, y_names, paths = load_dataset(args.data_dir, args.image_size, args.max_per_class)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_names)
    if len(encoder.classes_) != 2:
        raise ValueError("Exactly two classes are required for this initial version.")

    # Stratified split ensures both classes appear in each partition.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nDataset: {len(X)} images | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Classes (sorted): {encoder.classes_.tolist()}")
    print(f"Image features: {X.shape[1]} pixels")

    n_components = min(args.components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    print(f"PCA: {n_components} components retain {pca.explained_variance_ratio_.sum() * 100:.2f}% variance")

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(Z_train, y_train)
    lda_pred = lda.predict(Z_test)
    print("\n=== LDA evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, lda_pred):.4f} | Macro F1: {f1_score(y_test, lda_pred, average='macro'):.4f}")
    print(classification_report(y_test, lda_pred, target_names=encoder.classes_, zero_division=0))

    mlp = MLPClassifier(
        hidden_layer_sizes=(48, 24), activation="relu", solver="adam",
        alpha=0.0005, batch_size=64, learning_rate_init=0.001,
        max_iter=80, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=8, random_state=RANDOM_STATE, verbose=False
    )
    mlp.fit(Z_train, y_train)
    mlp_pred = mlp.predict(Z_test)
    print("=== Neural Network evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, mlp_pred):.4f} | Macro F1: {f1_score(y_test, mlp_pred, average='macro'):.4f}")
    print(classification_report(y_test, mlp_pred, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, mlp_pred))

    # Export the exact numerical parameters used by the browser inference code.
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    lda_coef = lda.coef_.tolist()
    lda_intercept = lda.intercept_.tolist()
    payload = {
        "project": "BuildSense AI",
        "model_version": 1,
        "image_size": args.image_size,
        "classes": encoder.classes_.tolist(),
        "cracked_class_index": int(next((i for i, name in enumerate(encoder.classes_) if "positive" in name.lower() or "crack" in name.lower()), 1)),
        "training": {
            "dataset_images_loaded": int(len(X)),
            "train_images": int(len(X_train)),
            "test_images": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "pca_components": int(n_components),
            "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
            "lda_test_accuracy": float(accuracy_score(y_test, lda_pred)),
            "lda_macro_f1": float(f1_score(y_test, lda_pred, average="macro")),
            "mlp_test_accuracy": float(accuracy_score(y_test, mlp_pred)),
            "mlp_macro_f1": float(f1_score(y_test, mlp_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_test, mlp_pred).tolist(),
        },
        "pca": {
            "mean": pca.mean_.astype(float).tolist(),
            "components": pca.components_.astype(float).tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "plot_range": [
                float(max(np.std(Z_train[:, 0]) * 3, 1e-6)),
                float(max(np.std(Z_train[:, 1]) * 3, 1e-6)) if Z_train.shape[1] > 1 else 1.0,
            ],
        },
        "lda": {"coef": lda_coef, "intercept": lda_intercept},
        "mlp": {
            "coefs": [w.astype(float).tolist() for w in mlp.coefs_],
            "intercepts": [b.astype(float).tolist() for b in mlp.intercepts_],
        },
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nExported browser model: {output} ({size_mb:.2f} MB)")
    print("Next: commit public/model.json, then deploy this static site on Vercel.")
    print("Reminder: results are experimental and are not a structural safety assessment.")


if __name__ == "__main__":
    main()
