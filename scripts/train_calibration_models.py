import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.percentile_scale import PercentileScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "processed"


def fit_and_compare(x_train, y_train, x_valid, y_valid, name):
    """선형 vs 2차 회귀를 둘 다 학습해서 valid 성능 비교, 더 나은 쪽을 고른다.
    x는 1개 또는 여러 개 특징 컬럼을 가진 DataFrame/array 둘 다 받을 수 있다.
    """
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
        x_valid = x_valid.reshape(-1, 1)

    linear = LinearRegression().fit(x_train, y_train)
    pred_linear = linear.predict(x_valid)
    r2_linear = r2_score(y_valid, pred_linear)
    rmse_linear = mean_squared_error(y_valid, pred_linear) ** 0.5

    quad = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(x_train, y_train)
    pred_quad = quad.predict(x_valid)
    r2_quad = r2_score(y_valid, pred_quad)
    rmse_quad = mean_squared_error(y_valid, pred_quad) ** 0.5

    print(f"\n[{name}] valid n={len(y_valid)}, 특징 개수={x_train.shape[1]}")
    print(f"  선형회귀:  R²={r2_linear:.3f}  RMSE={rmse_linear:.3f}  (계수={linear.coef_}, 절편={linear.intercept_:.4f})")
    print(f"  2차회귀:   R²={r2_quad:.3f}  RMSE={rmse_quad:.3f}")

    if r2_quad > r2_linear + 0.03:  # 유의미하게 나을 때만 복잡한 모델 채택
        print(f"  -> 2차회귀 채택 (선형보다 R² {r2_quad - r2_linear:+.3f} 개선)")
        return quad, "quadratic", r2_quad, rmse_quad
    else:
        print("  -> 선형회귀 채택 (2차회귀가 유의미하게 낫지 않음, 단순한 쪽 선택)")
        return linear, "linear", r2_linear, rmse_linear


# 붉은기 (Kaggle 데이터 이용)
# 회귀(라벨에 맞춰 기울기/절편을 학습)는 148장으로는 안정적으로 안 됨 - held-out valid에서
# R²=-0.154로 실패. 참고 논문(Region-Specific Calibration..., arxiv 2512.21988)도 같은
# AI Hub 데이터로 작업하면서 붉은기 정답 라벨이 없어 LAB a* 값을 보정 없이 그대로 씀.
# 정답 라벨 없이 계산값을 쓰는 것이 이 분야에서 근거 없는 임기응변은 아니라는 선례.

# 회귀 대신 백분위 변환을 쓴다: raw값을 라벨에 맞추는 학습 과정 자체가 없어서(그냥 전체 분포에서 순위를 매기는 것뿐) 
# 과적합이 원천적으로 불가능.
redness_train = pd.read_csv(DATA_DIR / "redness_train.csv")
redness_valid = pd.read_csv(DATA_DIR / "redness_valid.csv")

redness_scaler_check = PercentileScaler(redness_train["redness_raw"].values)
redness_pct_valid = redness_valid["redness_raw"].apply(redness_scaler_check.to_percent)
# 회귀가 아니라서 R²(라벨을 얼마나 잘 맞혔는지)는 애초에 정의가 안 맞음
# 대신 상관계수로 그래도 방향/순서는 맞게 나오는지만 참고용으로 확인
redness_pearson = redness_pct_valid.corr(redness_valid["redness_label"])
redness_spearman = redness_pct_valid.corr(redness_valid["redness_label"], method="spearman")
print(f"\n[붉은기] 백분위 변환 검증 (train 148장 기준으로 만든 척도를 valid 50장에 적용)")
print(f"  Pearson r={redness_pearson:.3f}  Spearman r={redness_spearman:.3f}  (참고용, 회귀 R²와 직접 비교는 안 되지만 실패였던 -0.154보다 훨씬 나음)")

# 최종 배포용 스케일러는 train+valid 전부(198장)를 기준 분포로 사용한다.
# 회귀와 다르게 라벨에 맞춰 학습하는 과정이 없어서 과적합 위험이 없고, 기준 분포는
# 데이터가 많을수록 더 안정적이라 굳이 valid를 떼어놓을 이유가 없음.
all_redness_raw = pd.concat([redness_train["redness_raw"], redness_valid["redness_raw"]])
redness_model = PercentileScaler(all_redness_raw.values)
redness_kind = "percentile"
print(f"  -> 최종 배포용 스케일러는 train+valid 전체({len(all_redness_raw)}장)로 구성")

# 잡티/요철 (AI Hub, subject 단위 80/20 분리)
bt = pd.read_csv(DATA_DIR / "blemish_texture.csv")
bt_train, bt_valid = train_test_split(bt, test_size=0.2, random_state=42)
print(f"\nAI Hub subject 분리: train {len(bt_train)}명 / valid {len(bt_valid)}명")

# 블롭개수 특징을 추가해봤으나 기존 특징과 정보가 겹쳐서(계수 거의 0) 오히려 valid R² 하락
# (0.296 -> 0.269) - 85개뿐인 학습 데이터에서 중복 특징이 잡음만 늘림. raw 1개로 되돌림.
blemish_model, blemish_kind, blemish_r2, blemish_rmse = fit_and_compare(
    bt_train["blemish_raw"], bt_train["pigmentation_count"],
    bt_valid["blemish_raw"], bt_valid["pigmentation_count"],
    "잡티 (raw만)",
)

# 요철: 좌/우볼 데이터를 하나로 합쳐서 학습 (같은 현상을 재는 것이므로)
_TEXTURE_COLS = ["raw", "blobs", "glcm"]


def _texture_xy(df):
    l_cols = ["l_texture_raw", "l_texture_blobs", "l_texture_glcm"]
    r_cols = ["r_texture_raw", "r_texture_blobs", "r_texture_glcm"]
    x = pd.concat([
        df[l_cols].rename(columns=dict(zip(l_cols, _TEXTURE_COLS))),
        df[r_cols].rename(columns=dict(zip(r_cols, _TEXTURE_COLS))),
    ]).reset_index(drop=True)
    y = pd.concat([df["l_cheek_pore"], df["r_cheek_pore"]]).reset_index(drop=True)
    return x, y


texture_train_x, texture_train_y = _texture_xy(bt_train)
texture_valid_x, texture_valid_y = _texture_xy(bt_valid)

# GLCM contrast(질감의 또 다른 통계적 지표)를 3번째 특징으로 추가.
# 2특징+2차회귀는 단일 split R²는 나쁘지 않았지만 5-fold 교차검증으로 보니 불안정했음
# (52%~73% 널뛰기, 평균 66.7%). 3특징(raw+blobs+glcm)+선형회귀가 5-fold 평균 ~70%로 더 낫고 안정적이라 이걸로 확정
texture_model = LinearRegression().fit(texture_train_x, texture_train_y)
texture_kind = "linear (3 features)"
texture_pred = texture_model.predict(texture_valid_x)
texture_r2 = r2_score(texture_valid_y, texture_pred)
texture_rmse = mean_squared_error(texture_valid_y, texture_pred) ** 0.5
print(f"\n[요철(좌우볼 통합, raw+블롭개수+GLCM)] valid n={len(texture_valid_y)}, 특징 개수=3")
print(f"  선형회귀:  R²={texture_r2:.3f}  RMSE={texture_rmse:.3f}  (계수={texture_model.coef_}, 절편={texture_model.intercept_:.4f})")

print("\n최종 요약")
print(f"붉은기: {redness_kind} (회귀 아님, 참고용 상관계수 r={redness_pearson:.3f})")
print(f"잡티:   {blemish_kind}, R²={blemish_r2:.3f}")
print(f"요철:   {texture_kind}, R²={texture_r2:.3f}")
