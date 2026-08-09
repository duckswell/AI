"""요철(피부 표면 거칠기) 원시 특징 계산 - v2 (해상도 정규화 추가).

v1은 사진 원본 해상도 그대로 라플라시안 분산을 계산했는데, 이 데이터셋은
사진마다 해상도가 다 달라서(Roboflow로 증강된 이미지 포함) 해상도 차이 자체가
분산값에 영향을 줬을 가능성이 높다. 실측 상관계수도 낮았다(r=0.15).

v2는 segmentation.normalize_face_crop()으로 먼저 크기를 통일한 크롭을 받는다는
전제로 동작한다 (호출부에서 정규화 후 넘겨줘야 함).
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def texture_raw(normalized_image_bgr: np.ndarray, normalized_mask: np.ndarray) -> float:
    """정규화된(고정 크기) 피부 크롭의 라플라시안 분산을 반환한다."""
    gray = cv2.cvtColor(normalized_image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    skin_values = laplacian[normalized_mask > 0]
    if skin_values.size == 0:
        raise ValueError("피부 마스크 영역이 비어있습니다.")

    return float(np.var(skin_values))


def texture_blob_count(normalized_image_bgr: np.ndarray, normalized_mask: np.ndarray,
                        blur_ksize: int = 15, top_percentile: float = 85, min_blob_area: int = 1) -> int:
    """모공은 라플라시안 분산(넓은 통계) 하나로는 잘 안 잡히는 '작고 많은 점'이라,
    잡티와 같은 방식(국소 대비 기반 블롭 검출)으로 모공 후보 개수를 세는 특징을 추가한다.
    모공은 잡티보다 훨씬 작고 촘촘해서 블러 커널을 더 작게, percentile은 더 낮게 잡았다.
    """
    gray = cv2.cvtColor(normalized_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    baseline_map = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    diff = baseline_map - gray

    skin_diff = diff[normalized_mask > 0]
    if skin_diff.size == 0:
        raise ValueError("피부 마스크 영역이 비어있습니다.")

    threshold = np.percentile(skin_diff, top_percentile)
    candidate_mask = ((diff >= threshold) & (normalized_mask > 0)).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    blob_areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(blob_areas >= min_blob_area))


def texture_glcm_contrast(normalized_image_bgr: np.ndarray, levels: int = 32) -> float:
    """GLCM(gray-level co-occurrence matrix) contrast - 라플라시안 분산/블롭개수와는 또 다른
    각도의 텍스처 지표. 인접 픽셀끼리 명암이 얼마나 급격히 바뀌는지를 통계적으로 잡아낸다.
    마스크를 따로 안 받는 이유: 이미 정규화된 크롭(피부 위주로 크롭된 200x200)이라 배경이
    거의 없어서, 크롭 전체에 대해 계산해도 큰 왜곡이 없다고 판단.
    """
    gray = cv2.cvtColor(normalized_image_bgr, cv2.COLOR_BGR2GRAY)
    quantized = (gray.astype(np.float32) / 256 * levels).astype(np.uint8)

    glcm = graycomatrix(quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels, symmetric=True, normed=True)
    return float(np.mean(graycoprops(glcm, "contrast")))
