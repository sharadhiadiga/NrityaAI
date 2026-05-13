"""
Train mudra RandomForest from folder images (same features as API).
Run: cd backend && python -m app.train_model --data-dir <DATASET_ROOT>
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.services.feature_extractor import FEATURE_VECTOR_SIZE
from app.services.mediapipe_service import MediaPipeHandService


def _image_extensions() -> set[str]:
    return {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_labeled_images(data_dir: Path):
    train_sub = data_dir / "train"
    base = train_sub if train_sub.is_dir() else data_dir
    if not base.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    exts = _image_extensions()
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        yield p.parent.name, p


def collect_features(
    data_dir: Path,
    service: MediaPipeHandService,
    max_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    rng = random.Random(seed)
    by_label: dict[str, List[Path]] = defaultdict(list)
    for label, path in iter_labeled_images(data_dir):
        by_label[label].append(path)

    if not by_label:
        raise SystemExit("No labeled images found under data directory.")

    work: List[Tuple[str, Path]] = []
    for label in sorted(by_label.keys()):
        paths = list(by_label[label])
        rng.shuffle(paths)
        paths = paths[:max_per_class]
        for p in paths:
            work.append((label, p))
    rng.shuffle(work)

    print("\nClass distribution (after per-class cap, before hand detection):")
    pre_counts = Counter(l for l, _ in work)
    for lab, c in sorted(pre_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {lab}: {c}")

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    skipped_no_hand = 0
    skipped_read = 0

    for label, img_path in work:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped_read += 1
            continue
        feat = service.feature_vector_from_bgr(bgr)
        if feat is None:
            skipped_no_hand += 1
            continue
        if feat.shape[0] != FEATURE_VECTOR_SIZE:
            raise RuntimeError(f"Feature dim mismatch: {feat.shape[0]} != {FEATURE_VECTOR_SIZE}")
        X_list.append(feat)
        y_list.append(label)
        if (len(X_list) % 250) == 0:
            print(f"  extracted {len(X_list)} samples…")

    print(f"\nSkipped (unreadable): {skipped_read}")
    print(f"Skipped (no hand): {skipped_no_hand}")
    print("\nClass distribution (usable rows):")
    use_counts = Counter(y_list)
    for lab, c in sorted(use_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {lab}: {c}")

    if not X_list:
        raise SystemExit("No usable samples after hand detection.")

    return np.stack(X_list, axis=0), y_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MUDRA_DATA_DIR", "data/mudra_dataset")),
        help="Dataset root (uses train/ if present).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=150,
        help="Max images per class (cap 150).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    max_per_class = max(1, min(args.max_per_class, 150))

    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "mudra_model.pkl"
    label_path = out_dir / "label_map.json"

    service = MediaPipeHandService()
    try:
        X, y = collect_features(data_dir, service, max_per_class=max_per_class, seed=args.seed)
    finally:
        service.close()

    counts = Counter(y)
    strat = y if counts and min(counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=strat,
        shuffle=True,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=13,
        min_samples_split=5,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    bundle = {"model": clf, "feature_dim": FEATURE_VECTOR_SIZE}
    joblib.dump(bundle, model_path)

    classes = [str(c) for c in clf.classes_.tolist()]
    label_map = {
        "classes": classes,
        "label_to_index": {lbl: i for i, lbl in enumerate(classes)},
        "index_to_label": {str(i): lbl for i, lbl in enumerate(classes)},
        "feature_dim": FEATURE_VECTOR_SIZE,
    }
    label_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"\nSaved model to {model_path}")
    print(f"Saved labels to {label_path}")


if __name__ == "__main__":
    main()
