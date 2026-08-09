"""3개 모델을 5-fold 교차검증으로 평가한다.

원래는 한 번만 train/valid로 나눠서 평가했는데, 표본이 작다보니(22~50개) 그 한 번의
나눔이 우연히 좋거나 나쁘게 나올 수 있다는 게 실제로 확인됐다 (잡티: 한 번은 68.2%,
5번 평균은 81.3% - 거의 13%p 차이). 그래서 5-fold 교차검증(전체 데이터를 5등분해서
번갈아가며 검증)으로 더 안정적인 평균치를 낸다.

예측/실제 모두 0~100% 스케일로 맞춘 뒤 하(0~50%)/상(50~100%) 2단계로 나눠서 일치율을 본다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.percentile_scale import PercentileScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "processed"


def tier2(pct: float) -> str:
    return "하" if pct < 50 else "상"


def kfold_accuracy(df, x_cols, y_col, name, n_splits=5, seed=1):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for train_idx, valid_idx in kf.split(df):
        train, valid = df.iloc[train_idx], df.iloc[valid_idx]
        model = LinearRegression().fit(train[x_cols], train[y_col])
        scaler = PercentileScaler(train[y_col].values)
        pred = pd.Series(model.predict(valid[x_cols])).apply(scaler.to_percent).apply(tier2).values
        actual = valid[y_col].apply(scaler.to_percent).apply(tier2).values
        accs.append((pred == actual).mean())
    print(f"{name}: {[f'{a:.1%}' for a in accs]} 평균 {np.mean(accs):.1%}")
    return float(np.mean(accs))


# 붉은기: 회귀가 아니라 백분위 변환이라 위 함수를 그대로 못 씀
redness = pd.concat([
    pd.read_csv(DATA_DIR / "redness_train.csv"),
    pd.read_csv(DATA_DIR / "redness_valid.csv"),
]).reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
redness_accs = []
for train_idx, valid_idx in kf.split(redness):
    train, valid = redness.iloc[train_idx], redness.iloc[valid_idx]
    raw_scaler = PercentileScaler(train["redness_raw"].values)
    label_scaler = PercentileScaler(train["redness_label"].values)
    pred = valid["redness_raw"].apply(raw_scaler.to_percent).apply(tier2).values
    actual = valid["redness_label"].apply(label_scaler.to_percent).apply(tier2).values
    redness_accs.append((pred == actual).mean())
print(f"붉은기: {[f'{a:.1%}' for a in redness_accs]} 평균 {np.mean(redness_accs):.1%}")
redness_acc = float(np.mean(redness_accs))

# 잡티
bt = pd.read_csv(DATA_DIR / "blemish_texture.csv")
blemish_acc = kfold_accuracy(bt, ["blemish_raw"], "pigmentation_count", "잡티")

# 요철: 사람 단위로 먼저 5등분한 뒤, 각 fold 안에서 좌/우볼을 행으로 풀어냄
# (좌/우볼을 먼저 풀고 나서 5등분하면 같은 사람의 왼쪽볼은 학습에, 오른쪽볼은 검증에
# 들어갈 수 있음 - 같은 사람 데이터가 양쪽에 섞이는 건 새어나감(leakage)에 가까워서 피함)
l_cols = ["l_texture_raw", "l_texture_blobs", "l_texture_glcm", "l_cheek_pore"]
r_cols = ["r_texture_raw", "r_texture_blobs", "r_texture_glcm", "r_cheek_pore"]
rename_to = ["raw", "blobs", "glcm", "cheek_pore"]


def _expand_lr(subset):
    return pd.concat([
        subset[l_cols].rename(columns=dict(zip(l_cols, rename_to))),
        subset[r_cols].rename(columns=dict(zip(r_cols, rename_to))),
    ]).reset_index(drop=True)


kf = KFold(n_splits=5, shuffle=True, random_state=1)
texture_accs = []
for train_idx, valid_idx in kf.split(bt):
    train = _expand_lr(bt.iloc[train_idx])
    valid = _expand_lr(bt.iloc[valid_idx])
    model = LinearRegression().fit(train[["raw", "blobs", "glcm"]], train["cheek_pore"])
    scaler = PercentileScaler(train["cheek_pore"].values)
    pred = pd.Series(model.predict(valid[["raw", "blobs", "glcm"]])).apply(scaler.to_percent).apply(tier2).values
    actual = valid["cheek_pore"].apply(scaler.to_percent).apply(tier2).values
    texture_accs.append((pred == actual).mean())
print(f"요철: {[f'{a:.1%}' for a in texture_accs]} 평균 {np.mean(texture_accs):.1%}")
texture_acc = float(np.mean(texture_accs))

print("\n최종 요약 (2단계 분류, 5-fold 평균)")
print(f"붉은기: {redness_acc:.1%}")
print(f"잡티:   {blemish_acc:.1%}")
print(f"요철:   {texture_acc:.1%}")
print("(참고: 2단계 랜덤 추측 시 기대 일치율은 약 50%)")
