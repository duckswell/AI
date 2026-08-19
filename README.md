<div align="center">
  <strong><img width="1920" height="1080" alt="중커톤 표지" src="https://github.com/user-attachments/assets/cd6d15f6-e577-458e-9b7d-8f580343eef2" />
</strong>
</div>

---

## 🧴 HALE 프로젝트 개요

### 서비스 소개
시술 후 피부 회복을 돕는 AI 스킨케어 코칭 앱

> **개발 기간**: 2026.07.26 ~ 2026.08.21

---

붉은기 / 요철 / 잡티 3개 지표를 사진에서 퍼센트로 산출하는 분석 로직.

---

## 🛠 기술 스택

<div align="center">

**Development**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Image Processing**

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=mediapipe&logoColor=white)

**Data / ML**

![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

</div>

---

## 🧪 최종 모델

| 지표 | 방식 |
|:---:|---|
| 붉은기 | 원시 특징값 → 기준 분포 백분위 변환 |
| 잡티 | 원시 특징값 → 선형회귀(스팟 개수 예측) → 백분위 변환 |
| 요철 | 원시 특징값 3종 → 선형회귀(모공 개수 예측) → 백분위 변환 |

---

## 구조

- `dataset/` - 학습/검증용 이미지 및 라벨 (gitignore, 출처는 `dataset/README.md` 참고)
- `src/features/` - 사진에서 원시 특징값을 계산하는 이미지 처리 로직
- `scripts/` - 데이터 정리(`build_processed_datasets.py`)부터 보정 모델 학습(`train_calibration_models.py`), export(`export_models.py`)까지의 실행 스크립트
- `models_export.json` - 최종 계수/기준 분포 (백엔드에서 로드할 파일)
- 자세한 개발 기록은 `notes/PROGRESS.md` (gitignore, 로컬 전용)
