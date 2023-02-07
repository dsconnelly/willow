import numpy as np

def butter(N: int, Wn: float, output: str, fs: float) -> np.ndarray:
    ...

def sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    ...