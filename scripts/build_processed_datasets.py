import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.segmentation import get_skin_mask, normalize_face_crop, imread_unicode
from src.features.redness import redness_raw
from src.features.blemish import blemish_raw
from src.features.texture import texture_raw, texture_blob_count, texture_glcm_contrast

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_redness(split: str, label_file: str):
    """Kaggle 데이터에서 붉은기 특징+라벨 추출."""
    dataset_dir = ROOT / "dataset" / "kaggle_skin_type"
    labels = pd.read_excel(dataset_dir / label_file)

    rows = []
    for _, row in labels.iterrows():
        image_id = row["Image_ID"]
        skin_type = image_id.split("_")[0]
        candidates = list((dataset_dir / split / skin_type).glob(f"{image_id}*"))
        if not candidates:
            continue
        image = imread_unicode(candidates[0])
        if image is None:
            continue
        mask = get_skin_mask(image)
        if mask is None:
            continue
        norm_image, norm_mask = normalize_face_crop(image, mask)
        rows.append({
            "image_id": image_id,
            "redness_raw": redness_raw(norm_image, norm_mask),
            "redness_label": row["Redness Severity (0-5)"],
        })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"redness_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{split}] 붉은기: {len(labels)}장 중 {len(df)}장 처리 완료 -> {out_path.name}")


def build_blemish_texture():
    """AI Hub 데이터에서 잡티/요철 특징+라벨 추출."""
    image_root = ROOT / "dataset" / "aihub_korean_skin" / "images"
    label_root = ROOT / "dataset" / "aihub_korean_skin" / "labels"

    rows = []
    subject_dirs = sorted(image_root.iterdir())
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        image_path = subject_dir / f"{subject_id}_01_F.jpg"
        label_dir = label_root / subject_id
        if not image_path.exists() or not label_dir.exists():
            continue

        try:
            part0 = json.load(open(label_dir / f"{subject_id}_01_F_00.json", encoding="utf-8"))
            part5 = json.load(open(label_dir / f"{subject_id}_01_F_05.json", encoding="utf-8"))
            part6 = json.load(open(label_dir / f"{subject_id}_01_F_06.json", encoding="utf-8"))
        except FileNotFoundError:
            continue

        pigmentation_count = part0["equipment"]["pigmentation_count"]
        l_cheek_pore = part5["equipment"].get("l_cheek_pore")
        r_cheek_pore = part6["equipment"].get("r_cheek_pore")
        if l_cheek_pore is None or r_cheek_pore is None:
            continue

        image = imread_unicode(image_path)
        if image is None:
            continue
        mask = get_skin_mask(image)
        if mask is None:
            continue

        blemish_value = blemish_raw(image, mask)

        l_x0, l_y0, l_x1, l_y1 = part5["images"]["bbox"]
        r_x0, r_y0, r_x1, r_y1 = part6["images"]["bbox"]
        l_crop = cv2.resize(image[l_y0:l_y1, l_x0:l_x1], (200, 200), interpolation=cv2.INTER_AREA)
        r_crop = cv2.resize(image[r_y0:r_y1, r_x0:r_x1], (200, 200), interpolation=cv2.INTER_AREA)
        full_mask = np.full((200, 200), 255, dtype=np.uint8)

        rows.append({
            "subject_id": subject_id,
            "blemish_raw": blemish_value,
            "pigmentation_count": pigmentation_count,
            "l_texture_raw": texture_raw(l_crop, full_mask),
            "l_texture_blobs": texture_blob_count(l_crop, full_mask),
            "l_texture_glcm": texture_glcm_contrast(l_crop),
            "l_cheek_pore": l_cheek_pore,
            "r_texture_raw": texture_raw(r_crop, full_mask),
            "r_texture_blobs": texture_blob_count(r_crop, full_mask),
            "r_texture_glcm": texture_glcm_contrast(r_crop),
            "r_cheek_pore": r_cheek_pore,
        })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "blemish_texture.csv"
    df.to_csv(out_path, index=False)
    print(f"잡티/요철: {len(subject_dirs)}명 중 {len(df)}명 처리 완료 -> {out_path.name}")


if __name__ == "__main__":
    build_redness("train", "labels_train.xlsx")
    build_redness("valid", "labels_valid.xlsx")
    build_blemish_texture()
