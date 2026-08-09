"""촬영 시점 품질 게이트.

조명에 따라 피부 측정값이 흔들리는 문제 자체를 완전히 없앨 순 없지만,
분석하기엔 너무 안 좋은 사진은 애초에 걸러서 사용자에게 다시 찍게 유도할 수 있다.
임계값은 기존 데이터셋 253장(둘 다 사람이 실제로 찍어서 쓸만하다고 판단된 사진들)의
밝기 분포(중앙값 152, 표준편차 18, 범위 102~208)를 참고해서, 정상 범위보다 확실히
벗어난 경우만 걸러지도록 여유 있게 잡았다.
"""

from src.features.segmentation import get_skin_mask
from src.features.reference_signals import brightness_evenness_hint

TOO_DARK_THRESHOLD = 60
TOO_BRIGHT_THRESHOLD = 230
UNEVEN_LIGHTING_THRESHOLD = 55  # brightness_std가 이보다 크면 얼굴 반쪽만 그림자/조명 쏠림


def check_capture_quality(image_bgr) -> dict:
    """사진이 분석하기에 적합한지 판정한다.

    반환: {"ok": bool, "reason": str|None} - reason은 ok가 False일 때 사용자에게 보여줄 안내 문구용 코드.
    """
    mask = get_skin_mask(image_bgr)
    if mask is None:
        return {"ok": False, "reason": "no_face_detected"}

    stats = brightness_evenness_hint(image_bgr, mask)

    if stats["mean_brightness"] < TOO_DARK_THRESHOLD:
        return {"ok": False, "reason": "too_dark"}
    if stats["mean_brightness"] > TOO_BRIGHT_THRESHOLD:
        return {"ok": False, "reason": "too_bright"}
    if stats["brightness_std"] > UNEVEN_LIGHTING_THRESHOLD:
        return {"ok": False, "reason": "uneven_lighting"}

    return {"ok": True, "reason": None}
