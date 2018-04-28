from collections import Counter
from typing import Dict

import numpy as np

from colors import ColorT, COLOR_2_TYPE
from split_generator import generate_chunks_from_file


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
    return pixel_sum / pixel_cnt


def extract_features(img_path: str,
                     mask_path: str,
                     out_path: str,
                     chunk_size: int = 4,
                     ) -> None:
    img_chunks = generate_chunks_from_file(
        img_path,
        size_x=chunk_size,
        size_y=chunk_size,
        step_x=chunk_size,
        step_y=chunk_size
    )

    mask_chunks = generate_chunks_from_file(
        mask_path,
        size_x=chunk_size,
        size_y=chunk_size,
        step_x=chunk_size,
        step_y=chunk_size
    )
    chunks = zip(img_chunks, mask_chunks)

    with open(out_path, 'a') as file:
        for img_chunk, mask_chunk in chunks:
            chunk_t = chunk_type(mask_chunk)
            chunk_desc1, chunk_desc2, chunk_desc3 = chunk_descriptor(img_chunk)
            file.write('{},{},{},{}\n'.format(chunk_t, chunk_desc1, chunk_desc2, chunk_desc3))


if __name__ == '__main__':
    # img = np.array(
    #     [
    #         [[10, 10, 0], [20, 20, 20]],
    #         [[5, 5, 5], [5, 5, 5]]
    #     ]
    # )
    # mask = np.array(
    #     [
    #         [[224, 224, 224], [224, 224, 224]],
    #         [[255, 128, 0], [224, 224, 224]]
    #     ]
    # )
    # print(chunk_descriptor(img))
    # print(chunk_type(mask))
    extract_features('data/water/00.32953.jpg', 'data/water/00.32953_mask.png', 'out/features.csv')
