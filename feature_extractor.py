from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from colors import ColorT, COLOR_2_TYPE


def chunk_type(mask_chunk: np.ndarray,
               color2type: Dict[ColorT, int] = COLOR_2_TYPE) -> int:
    counter = Counter()

    for row in mask_chunk:
        for pixel in row:
            counter[tuple(pixel)] += 1

    most_recent_color = counter.most_common(1)[0][0]
    return color2type[most_recent_color]


def chunk_descriptor(img_chunk: np.ndarray) -> np.ndarray:
    height, width, _ = img_chunk.shape
    pixel_sum = np.sum(img_chunk, axis=(0, 1), dtype=np.int)
    pixel_cnt = height * width
    return pixel_sum // pixel_cnt


def extract_features(img: np.ndarray,
                     mask: np.ndarray,
                     chunk_size: int = 4) -> List[Tuple[int, np.ndarray]]:
    pass


if __name__ == '__main__':
    img = np.array(
        [
            [[10, 10, 0], [20, 20, 20]],
            [[5, 5, 5], [5, 5, 5]]
        ]
    )
    mask = np.array(
        [
            [[224, 224, 224], [224, 224, 224]],
            [[255, 128, 0], [224, 224, 224]]
        ]
    )
    print(chunk_descriptor(img))
    print(chunk_type(mask))
