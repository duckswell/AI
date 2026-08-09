"""사진에서 피부 영역(배경/머리카락/눈/눈썹/입 제외)만 골라내는 마스크를 만든다.
붉은기/요철/잡티 등 이후의 모든 특징 계산은 이 마스크를 공통으로 사용한다.
"""

import cv2
import numpy as np
import mediapipe as mp


def imread_unicode(path) -> np.ndarray | None:
    """cv2.imread는 Windows에서 한글 등 non-ASCII 경로를 못 읽고 조용히 None을 반환한다.
    파일을 바이트로 직접 읽어서 디코딩하는 방식으로 우회한다.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

_mp_face_mesh = mp.solutions.face_mesh

# mediapipe FaceMesh landmark index sets (refine_landmarks=True 기준)
_FACE_OVAL = list({i for pair in _mp_face_mesh.FACEMESH_FACE_OVAL for i in pair})
_LEFT_EYE = list({i for pair in _mp_face_mesh.FACEMESH_LEFT_EYE for i in pair})
_RIGHT_EYE = list({i for pair in _mp_face_mesh.FACEMESH_RIGHT_EYE for i in pair})
_LEFT_EYEBROW = list({i for pair in _mp_face_mesh.FACEMESH_LEFT_EYEBROW for i in pair})
_RIGHT_EYEBROW = list({i for pair in _mp_face_mesh.FACEMESH_RIGHT_EYEBROW for i in pair})
_LIPS = list({i for pair in _mp_face_mesh.FACEMESH_LIPS for i in pair})


def _landmarks_to_points(landmarks, indices, width, height):
    pts = []
    for i in indices:
        lm = landmarks[i]
        pts.append([int(lm.x * width), int(lm.y * height)])
    return np.array(pts, dtype=np.int32)


def _fill_convex(mask, points, value):
    if len(points) < 3:
        return
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, value)


def get_skin_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """얼굴을 찾아 피부 영역만 255인 마스크(uint8, HxW)를 반환한다.
    얼굴을 찾지 못하면 None을 반환한다.
    """
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    mask = np.zeros((height, width), dtype=np.uint8)
    face_pts = _landmarks_to_points(landmarks, _FACE_OVAL, width, height)
    _fill_convex(mask, face_pts, 255)

    for indices in (_LEFT_EYE, _RIGHT_EYE, _LEFT_EYEBROW, _RIGHT_EYEBROW, _LIPS):
        exclude_pts = _landmarks_to_points(landmarks, indices, width, height)
        _fill_convex(mask, exclude_pts, 0)

    return mask


def apply_skin_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 바깥 영역을 검게 지운 이미지를 반환한다 (시각적 확인용)."""
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def skin_pixel_ratio(mask: np.ndarray) -> float:
    """전체 이미지 대비 피부로 판정된 픽셀의 비율 (0~1)."""
    return float(np.count_nonzero(mask)) / mask.size


def normalize_face_crop(image_bgr: np.ndarray, mask: np.ndarray, size: int = 300):
    """얼굴 영역을 기준으로 크롭하고 고정 크기로 리사이즈한다.

    사진마다 해상도/얼굴 크기가 달라서, 특히 텍스처처럼 픽셀 단위 변화량에
    민감한 특징은 해상도 자체가 값에 영향을 준다. 크기를 통일해서 이 영향을 없앤다.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("피부 마스크 영역이 비어있습니다.")

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    cropped_image = image_bgr[y0:y1 + 1, x0:x1 + 1]
    cropped_mask = mask[y0:y1 + 1, x0:x1 + 1]

    resized_image = cv2.resize(cropped_image, (size, size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(cropped_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    return resized_image, resized_mask
