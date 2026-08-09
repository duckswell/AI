"""잡티(반점) 원시 특징 계산 - v2 (얼굴 내 상대값 기준, 붉은기와 같은 방식).

v1(고정 임계값으로 반점 픽셀 비율)은 사람마다 사진 노출/대비가 달라서
실측 상관관계가 거의 없었고 심지어 방향이 반대로 나왔다(r=-0.13).

v2는 '국소 평균보다 어두운 정도'의 상위 percentile 값이, 그 얼굴의 중앙값보다
얼마나 튀는지를 본다 (redness_raw와 동일한 설계 원칙 - 얼굴별 상대 기준).
"""

import cv2
import numpy as np


def blemish_raw(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    blur_ksize: int = 31,
    top_percentile: float = 90,
) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    baseline_map = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    diff = baseline_map - gray  # 국소 베이스보다 어두우면 양수

    skin_diff = diff[mask > 0]
    if skin_diff.size == 0:
        raise ValueError("피부 마스크 영역이 비어있습니다.")

    baseline = float(np.median(skin_diff))
    threshold = np.percentile(skin_diff, top_percentile)
    darkest = skin_diff[skin_diff >= threshold]

    return float(np.mean(darkest)) - baseline
