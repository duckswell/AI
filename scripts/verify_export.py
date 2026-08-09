"""export한 JSON만으로(학습 코드 없이) 실제 사진 몇 장을 끝까지 처리해서, Java가 이 JSON을
가지고 그대로 구현했을 때 나올 최종 결과와 같은 값이 나오는지 확인한다.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.segmentation import get_skin_mask, normalize_face_crop, imread_unicode
from src.features.redness import redness_raw
from src.features.blemish import blemish_raw
from src.features.texture import texture_raw, texture_blob_count, texture_glcm_contrast

ROOT = Path(__file__).resolve().parent.parent
export = json.load(open(ROOT / "models_export.json", encoding="utf-8"))


def percentile_of(value, reference_percentiles):
    """reference_percentiles(0~100 지점의 값 101개) 안에서 value가 몇 %에 해당하는지 보간."""
    qs = np.linspace(0, 100, len(reference_percentiles))
    return float(np.interp(value, reference_percentiles, qs))


def predict_redness(raw):
    return percentile_of(raw, export["redness"]["reference_percentiles"])


def predict_blemish(raw):
    coef = export["blemish"]["regression"]["coefficients"][0]
    intercept = export["blemish"]["regression"]["intercept"]
    predicted_count = coef * raw + intercept
    return percentile_of(predicted_count, export["blemish"]["reference_percentiles"])


def predict_texture(raw, blobs, glcm):
    coefs = export["texture"]["regression"]["coefficients"]
    intercept = export["texture"]["regression"]["intercept"]
    # coefficients 순서: [raw, blobs, glcm]
    predicted_count = sum(c * f for c, f in zip(coefs, [raw, blobs, glcm])) + intercept
    return percentile_of(predicted_count, export["texture"]["reference_percentiles"])


# 샘플 사진 하나로 전체 파이프라인 확인
sample_path = ROOT / "dataset" / "aihub_korean_skin" / "images" / "0001" / "0001_01_F.jpg"
image = imread_unicode(sample_path)
mask = get_skin_mask(image)
norm_image, norm_mask = normalize_face_crop(image, mask)

r = redness_raw(norm_image, norm_mask)
b = blemish_raw(image, mask)
t_raw = texture_raw(norm_image, norm_mask)
t_blobs = texture_blob_count(norm_image, norm_mask)
t_glcm = texture_glcm_contrast(norm_image)

print(f"샘플: {sample_path.parent.name}")
print(f"붉은기: raw={r:.2f} -> {predict_redness(r):.1f}%")
print(f"잡티:   raw={b:.2f} -> {predict_blemish(b):.1f}%")
print(f"요철:   raw={t_raw:.2f}, blobs={t_blobs}, glcm={t_glcm:.2f} -> {predict_texture(t_raw, t_blobs, t_glcm):.1f}%")
