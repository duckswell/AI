"""프로덕션에서 Backend가 촬영 직후 호출할 가벼운 사진 품질 체크 스크립트.

analyze.py의 무거운 특징 추출 없이, 이 사진이 분석하기에 적합한지만 빠르게 판정한다.

사용법: python check_photo.py <image_path>
출력: {"ok": bool, "reason": str|None} - reason은 ok가 False일 때 사용자 안내용 코드.
exit code: ok가 True면 0, False면 1.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.segmentation import imread_unicode
from src.features.capture_quality import check_capture_quality


def check(image_path: str) -> dict:
    if not Path(image_path).is_file():
        return {"ok": False, "reason": "image_not_found"}

    image = imread_unicode(image_path)
    if image is None:
        return {"ok": False, "reason": "image_not_found"}
    return check_capture_quality(image)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "reason": "usage: check_photo.py <image_path>"}, ensure_ascii=False))
        sys.exit(1)

    result = check(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("ok") else 1)
