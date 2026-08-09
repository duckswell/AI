"""붉은기(erythema) 원시 특징 계산 - v2 (얼굴 내 상대값 기준).

v1(전체 평균 a*)은 실측 검증 결과 사람 라벨과 상관관계가 거의 없었다(r=0.08).
원인으로 추정되는 건 이 데이터셋 사진들이 임상 사진과 달리 조명/화이트밸런스가
제각각이라, 사진 전체가 통째로 붉거나 노랗게 치우쳐버리는 노이즈가 실제 붉은기
차이보다 커서 신호를 덮어버린다는 것.

v2는 '그 얼굴 안에서 가장 붉은 부분이, 그 얼굴 평균보다 얼마나 더 붉은가'를 본다.
조명이 얼굴 전체를 다 붉게 만들어도 평균과 극단값이 같이 움직이니 상쇄되고,
실제로 국소적으로 붉은 부위(뺨 등)만 있을 때 그 차이가 남는다.
"""

import cv2
import numpy as np


def redness_raw(image_bgr: np.ndarray, mask: np.ndarray, top_percentile: float = 90) -> float:
    """(상위 percentile 붉은 픽셀들의 평균 a*) - (전체 피부 중앙값 a*)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float64)

    skin_values = a_channel[mask > 0]
    if skin_values.size == 0:
        raise ValueError("피부 마스크 영역이 비어있습니다.")

    baseline = float(np.median(skin_values))
    threshold = np.percentile(skin_values, top_percentile)
    reddest = skin_values[skin_values >= threshold]

    return float(np.mean(reddest)) - baseline
