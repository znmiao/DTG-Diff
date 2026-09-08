from __future__ import annotations

import numpy as np


def subsequences(array: np.ndarray, window_size: int, time_step: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if time_step <= 0:
        raise ValueError("time_step must be positive")

    arr = np.asarray(array)
    if arr.shape[0] < window_size:
        shape = (0, window_size) + arr.shape[1:]
        return np.empty(shape, dtype=arr.dtype)

    starts = range(0, arr.shape[0] - window_size + 1, time_step)
    return np.stack([arr[s:s + window_size] for s in starts], axis=0)
