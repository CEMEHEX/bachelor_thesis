import ntpath
import os
import pickle
import re
from os.path import exists, isfile, join
from random import shuffle
from typing import Dict, List, Tuple, Optional, Generator

import cv2
import numpy as np

from feature_extractor import chunk_type, chunk_descriptor
from split_generator import generate_chunks_and_positions_from_file
from utils import prepare_environment, get_data_paths, get_name, create_if_not_exists, shuffle_csv


class Stats:
    def __init__(self,
                 img_name: str,
                 mask_name: str,
                 surf_info: Dict[int, List[Tuple[int, int]]],
                 chunk_size: int
                 ):
        self.img_name = img_name
        self.mask_name = mask_name
        self.surf_info = surf_info
        self.chunk_size = chunk_size

    def get_chunks(self,
                   surf_type: int,
                   cnt: Optional[int] = None) -> List[np.ndarray]:
        desc = self.surf_info[surf_type]
        if cnt is None:
            cnt = len(desc)
        assert cnt <= len(desc)
        img = cv2.imread(self.img_name)

        return [img[y:y + self.chunk_size, x:x + self.chunk_size]
                for x, y in desc[0:cnt]]

    def min_chunks_cnt(self) -> int:
        surf_descriptions = self.surf_info.values()
        chunk_count = [len(desc) for desc in surf_descriptions if len(desc) > 0]
        return min(chunk_count)

    def get_balanced_chunk_positions(self, size: float = 1.0) -> Generator[np.ndarray, None, None]:
        img = cv2.imread(self.img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.imread(self.mask_name)
        positions_list = [desc[0:self.min_chunks_cnt()]
                          for desc in self.surf_info.values()]
        positions = [pos for poss in positions_list for pos in poss]  # concatenation of all positions lists
        shuffle(positions)
        positions = positions[0:int(len(positions) * size)]

        for x, y in positions:
            yield img[y:y + self.chunk_size, x:x + self.chunk_size], \
                  mask[y:y + self.chunk_size, x:x + self.chunk_size]


def read_stats_from_file(filename: str) -> Stats:
    with open(filename, 'rb') as file:
        return pickle.load(file)


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

    surf_info: Dict[int, List[Tuple[int, int]]] = {}
    for x, y, img_chunk, mask_chunk in data:
        chunk_t = chunk_type(mask_chunk)
        cur_info = surf_info.get(chunk_t, [])
        cur_info.append((x, y))
        surf_info[chunk_t] = cur_info

    for info in surf_info.values():
        shuffle(info)

    return Stats(img_filename, mask_filename, surf_info, chunk_size)


def calc_and_save_stats(img_filename: str,
                        mask_filename: str,
                        output_filename: str,
                        chunk_size: int) -> None:
    stats = calc_stats(img_filename, mask_filename, chunk_size)
    with open(output_filename, 'wb') as file:
        pickle.dump(stats, file, protocol=pickle.HIGHEST_PROTOCOL)


def calc_dataset_stats(dataset_name: str) -> None:
    paths = get_data_paths(f'data/{dataset_name}')
    create_if_not_exists(f'statistics/{dataset_name}')

    for img_path, mask_path in paths:
        output_path = f'statistics/{dataset_name}/{ntpath.basename(get_name(img_path))}.pickle'
        calc_and_save_stats(
            img_filename=img_path,
            mask_filename=mask_path,
            output_filename=output_path,
            chunk_size=4
        )


def extract_features_using_stats(dataset_name: str,
                                 size: float = 1.0):
    out_path = f'out/{dataset_name}_features.csv'
    if exists(out_path):
        os.remove(out_path)

    data_path = f'data/{dataset_name}'
    statistics_path = f'statistics/{dataset_name}'

    filenames = [f for f in os.listdir(statistics_path)]
    filenames = [re.sub('\.pickle', '', f) for f in filenames]
    print(filenames)

    for filename in filenames:
        stats = read_stats_from_file(f'statistics/{dataset_name}/{filename}.pickle')
        chunks = stats.get_balanced_chunk_positions(size)

        with open(out_path, 'a') as file:
            for img_chunk, mask_chunk in chunks:
                chunk_t = chunk_type(mask_chunk)
                chunk_desc1, chunk_desc2, chunk_desc3 = chunk_descriptor(img_chunk)
                file.write(f'{chunk_t},{chunk_desc1},{chunk_desc2},{chunk_desc3}\n')

        shuffle_csv(out_path)


if __name__ == '__main__':
    prepare_environment()
    # stats = calc_stats('data/water/00.32953.jpg', 'data/water/00.32953_mask.png', 4)
    # stats = read_stats_from_file('out/stats_sample.pickle')


    # chunks = stats.get_chunks(7, 50)
    #
    # print('Done')
    # view_images([chunks], ['kek'])

    # calc_and_save_stats(
    #     img_filename='data/water/00.32953.jpg',
    #     mask_filename='data/water/00.32953_mask.png',
    #     output_filename='out/stats_sample.pickle',
    #     chunk_size=4)

    calc_dataset_stats('old_methods_dataset')
    extract_features_using_stats('old_methods_dataset')
