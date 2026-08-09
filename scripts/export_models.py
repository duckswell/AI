import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "processed"
OUT_PATH = ROOT / "models_export.json"


def percentile_breakpoints(values, n=101):
    """0~100 백분위 지점의 값들을 리스트로 저장"""
    qs = np.linspace(0, 100, n)
    return [float(v) for v in np.percentile(values, qs)]


export = {}

#붉은기: 백분위 변환
redness_train = pd.read_csv(DATA_DIR / "redness_train.csv")
redness_valid = pd.read_csv(DATA_DIR / "redness_valid.csv")
all_redness_raw = pd.concat([redness_train["redness_raw"], redness_valid["redness_raw"]])

export["redness"] = {
    "method": "percentile",
    "input": "redness_raw",
    "reference_percentiles": percentile_breakpoints(all_redness_raw.values),
}

#잡티/요철: 회귀(raw -> 개수) + 백분위(개수 -> %)
bt = pd.read_csv(DATA_DIR / "blemish_texture.csv")

blemish_model = LinearRegression().fit(bt[["blemish_raw"]], bt["pigmentation_count"])
blemish_pred_all = blemish_model.predict(bt[["blemish_raw"]])

export["blemish"] = {
    "method": "linear_then_percentile",
    "input": "blemish_raw",
    "regression": {
        "coefficients": blemish_model.coef_.tolist(),
        "intercept": float(blemish_model.intercept_),
        "predicts": "pigmentation_count",
    },
    # 회귀가 예측한 '개수'를 다시 관측된 개수 분포 기준 백분위로 변환하기 위한 기준값
    "reference_percentiles": percentile_breakpoints(bt["pigmentation_count"].values),
}

# raw+blobs 2특징+2차회귀는 5-fold 교차검증에서 불안정(52~73% 널뛰기, 평균 66.7%)해서,
# GLCM contrast를 3번째 특징으로 추가하고 선형회귀로 바꿈 - 5-fold 평균 ~70%로 개선 및 안정화.
_TEXTURE_COLS = ["raw", "blobs", "glcm"]
l_cols = ["l_texture_raw", "l_texture_blobs", "l_texture_glcm"]
r_cols = ["r_texture_raw", "r_texture_blobs", "r_texture_glcm"]
texture_x = pd.concat([
    bt[l_cols].rename(columns=dict(zip(l_cols, _TEXTURE_COLS))),
    bt[r_cols].rename(columns=dict(zip(r_cols, _TEXTURE_COLS))),
]).reset_index(drop=True)
texture_y = pd.concat([bt["l_cheek_pore"], bt["r_cheek_pore"]]).reset_index(drop=True)

texture_model = LinearRegression().fit(texture_x, texture_y)

export["texture"] = {
    "method": "linear_then_percentile",
    "input": ["texture_raw", "texture_blob_count", "texture_glcm_contrast"],
    "regression": {
        "coefficients": texture_model.coef_.tolist(),
        "intercept": float(texture_model.intercept_),
        "predicts": "pore_count",
        "note": "coefficients 순서대로 [raw, blobs, glcm] 특징에 대응",
    },
    "reference_percentiles": percentile_breakpoints(texture_y.values),
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {OUT_PATH}")
print(f"파일 크기: {OUT_PATH.stat().st_size:,} bytes")
