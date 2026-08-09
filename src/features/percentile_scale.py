import numpy as np


class PercentileScaler:
    def __init__(self, reference_values: np.ndarray):
        self.reference_sorted = np.sort(np.asarray(reference_values))

    def to_percent(self, raw_value: float) -> float:
        """reference_values 분포 안에서 raw_value가 몇 번째 백분위인지 반환 (0~100)."""
        rank = np.searchsorted(self.reference_sorted, raw_value, side="right")
        return float(rank) / len(self.reference_sorted) * 100
