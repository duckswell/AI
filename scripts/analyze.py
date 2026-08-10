"""프로덕션에서 Backend가 호출할 최종 분석 스크립트.

사진 한 장을 받아 붉은기/잡티/요철 3개 지표를 models_export.json의
캘리브레이션 값으로 변환해 JSON으로 stdout에 출력한다.

사용법: python analyze.py <image_path>
성공: {"redness_pct": .., "blemish_pct": .., "texture_pct": ..} 출력, exit 0
실패: {"error": "image_not_found" | "no_face_detected"} 출력, exit 1
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
EXPORT = json.load(open(ROOT / "models_export.json", encoding="utf-8"))


def _percentile_of(value, reference_percentiles):
    """reference_percentiles(0~100 지점의 값 101개) 안에서 value가 몇 %에 해당하는지 보간."""
    qs = np.linspace(0, 100, len(reference_percentiles))
    return float(np.interp(value, reference_percentiles, qs))


def _predict_redness(raw):
    return _percentile_of(raw, EXPORT["redness"]["reference_percentiles"])


def _predict_blemish(raw):
    coef = EXPORT["blemish"]["regression"]["coefficients"][0]
    intercept = EXPORT["blemish"]["regression"]["intercept"]
    predicted_count = coef * raw + intercept
    return _percentile_of(predicted_count, EXPORT["blemish"]["reference_percentiles"])


def _predict_texture(raw, blobs, glcm):
    coefs = EXPORT["texture"]["regression"]["coefficients"]
    intercept = EXPORT["texture"]["regression"]["intercept"]
    # coefficients 순서: [raw, blobs, glcm]
    predicted_count = sum(c * f for c, f in zip(coefs, [raw, blobs, glcm])) + intercept
    return _percentile_of(predicted_count, EXPORT["texture"]["reference_percentiles"])


def analyze(image_path: str) -> dict:
    if not Path(image_path).is_file():
        return {"error": "image_not_found"}

    image = imread_unicode(image_path)
    if image is None:
        return {"error": "image_not_found"}

    mask = get_skin_mask(image)
    if mask is None:
        return {"error": "no_face_detected"}

    norm_image, norm_mask = normalize_face_crop(image, mask)

    r = redness_raw(norm_image, norm_mask)
    b = blemish_raw(image, mask)
    t_raw = texture_raw(norm_image, norm_mask)
    t_blobs = texture_blob_count(norm_image, norm_mask)
    t_glcm = texture_glcm_contrast(norm_image)

    return {
        "redness_pct": round(_predict_redness(r), 1),
        "blemish_pct": round(_predict_blemish(b), 1),
        "texture_pct": round(_predict_texture(t_raw, t_blobs, t_glcm), 1),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: analyze.py <image_path>"}, ensure_ascii=False))
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(1 if "error" in result else 0)
