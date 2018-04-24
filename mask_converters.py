from typing import Tuple

import numpy as np

WHITE_COL = 255
BLACK_COL = 0
WATER_COL = (255, 128, 0)

ColorT = Tuple[int, int, int]


def convert_to_binary(mask: np.ndarray, primary_color: ColorT) -> np.ndarray:
    res = np.zeros(mask.shape[0:2], dtype=np.uint8)
    res[np.where((mask == primary_color).all(axis=2))[0:2]] = WHITE_COL
    return res


def convert_to_binary_water(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=WATER_COL)
