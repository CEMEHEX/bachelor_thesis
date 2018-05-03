import pickle
from random import shuffle
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from feature_extractor import chunk_type
from split_generator import generate_chunks_and_positions_from_file
from utils import view_images


class SurfDescription:
    def __init__(self,
                 chunk_positions: List[Tuple[int, int]],
                 chunk_size: int):
        self.chunk_positions = chunk_positions
        self.chunk_size = chunk_size

    def chunks_cnt(self) -> int:
        return len(self.chunk_positions)


class Stats:
    def __init__(self,
                 img_name: str,
                 mask_name: str,
                 surf_info: Dict[int, SurfDescription]
                 ):
        self.img_name = img_name
        self.mask_name = mask_name
        self.surf_info = surf_info

    def get_chunks(self,
                   surf_type: int,
                   cnt: Optional[int] = None) -> List[np.ndarray]:
        desc = self.surf_info[surf_type]
        if cnt is None:
            cnt = desc.chunks_cnt()
        assert cnt <= desc.chunks_cnt()
        img = cv2.imread(self.img_name)

        return [img[y:y + desc.chunk_size, x:x + desc.chunk_size]
                for x, y in desc.chunk_positions[0:cnt]]


def read_stats_from_file(filename: str) -> Stats:
    return pickle.load(filename)


def calc_stats(img_filename: str,
               mask_filename: str,
               chunk_size: int) -> Stats:
    data = generate_chunks_and_positions_from_file(
        img_path=img_filename,
        mask_path=mask_filename,
        size_x=chunk_size,
        size_y=chunk_size,
        step_x=chunk_size,
        step_y=chunk_size
    )

    surf_info: Dict[int, SurfDescription] = {}
    cnt = 0
    for x, y, img_chunk, mask_chunk in data:
        chunk_t = chunk_type(mask_chunk)
        cur_info = surf_info.get(chunk_t, SurfDescription([], chunk_size))
        cur_info.chunk_positions.append((x, y))
        surf_info[chunk_t] = cur_info

    for info in surf_info.values():
        shuffle(info.chunk_positions)

    return Stats(img_filename, mask_filename, surf_info)


def calc_and_save_stats(img_filename: str,
                        mask_filename: str,
                        output_filename: str,
                        chunk_size: int) -> None:
    stats = calc_stats(img_filename, mask_filename, chunk_size)
    pickle.dump(stats, output_filename)


if __name__ == '__main__':
    stats = calc_stats('data/water/00.32953.jpg', 'data/water/00.32953_mask.png', 4)
    chunks = stats.get_chunks(2, 50)

    print('Done')
    view_images([chunks], ['kek'])