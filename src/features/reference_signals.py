"""LLM 프롬프트에 참고자료로 얹을 부가 신호.

원래 유분기(하이라이트 비율)도 여기 있었는데, Kaggle 폴더 라벨(oily/dry/normal)로
방향성만이라도 확인해보니 oily가 오히려 제일 낮게 나와서(0.018 < dry 0.035) 뺐다.
정답 라벨이 아예 없는 상태에서 방향마저 이상하면 LLM한테 혼란만 준다고 판단.
"""

import cv2
import numpy as np


def brightness_evenness_hint(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """피부 영역의 평균 밝기와 밝기 표준편차. 표준편차가 크면 얼굴 안에서 밝은 곳/어두운 곳 차이가 크다는 뜻."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    skin_values = gray[mask > 0].astype(np.float64)
    if skin_values.size == 0:
        raise ValueError("피부 마스크 영역이 비어있습니다.")

    return {
        "mean_brightness": float(np.mean(skin_values)),
        "brightness_std": float(np.std(skin_values)),
    }
