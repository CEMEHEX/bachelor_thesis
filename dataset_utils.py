import itertools
import pickle
from collections import defaultdict
from random import shuffle
from typing import Callable, Set, List, Dict

import cv2
import numpy as np

from colors import ColorT
from mask_converters import identity, TO_BIN_CONVERTERS, CLASS_TO_COL
from split_generator import dataset_generator
from utils import get_data_paths, files_cnt, clear_and_create

# The n-th element of the list represents a dictionary that
# for each color stores the percentage of this color in the n-th mask
StatisticsT = List[Dict[ColorT, float]]


class DatasetInfo:
    def __init__(
            self,
            statistics: StatisticsT,
            class_positions: Dict[ColorT, Set[int]]
    ):
        self.statistics = statistics
        self.class_positions = class_positions

    # returns set of numbers of masks which contains target class and other classes
    def get_mixed(self, col) -> Set[int]:
        target_class_positions = self.class_positions[col]
        other_classes_positions = {c: self.class_positions[c]
                                   for c in self.class_positions
                                   if c != col}
        intersections = []
        for c in other_classes_positions:
            class_positions = other_classes_positions[c]
            intersections.append(target_class_positions & class_positions)

        result = set().union(*intersections)
        return result

    # returns set of indices of masks which contains other classes
    def others_union(self, col) -> Set[int]:
        other_classes_positions = {c: self.class_positions[c]
                                   for c in self.class_positions
                                   if c != col}
        others_union = set().union(*other_classes_positions.values())
        return others_union

    # returns set of indices of masks which contains ONLY target class
    def get_pure(self, col) -> Set[int]:
        others_union = self.others_union(col)
        return self.class_positions[col] - others_union

    # returns set of indices of masks which contains ONLY other classes
    def get_others(self, col) -> Set[int]:
        others_union = self.others_union(col)
        return others_union - self.class_positions[col]


def get_info(dataset_path: str) -> DatasetInfo:
    statistics = []
    class_positions = defaultdict(set)

    n = files_cnt(dataset_path)
    assert n % 2 == 0
    n //= 2

    for i in range(n):
        print('processing', '{}/{}_mask.png'.format(dataset_path, i))
        cur_mask = cv2.imread('{}/{}_mask.png'.format(dataset_path, i))
        cur_stats = get_mask_stats(cur_mask)
        statistics.append(cur_stats)

        for col in cur_stats:
            class_positions[col].add(i)

    return DatasetInfo(statistics, class_positions)


def get_mask_stats(mask: np.ndarray) -> Dict[ColorT, float]:
    res = defaultdict(float)

    height, width, _ = mask.shape
    for row in mask:
        for pixel in row:
            res[tuple(pixel)] += 1

    for col in res:
        res[col] /= height * width

    return res


def calc_and_save_info(crops_folder_name: str):
    dataset_path = 'data/{}'.format(crops_folder_name)
    output_filename = 'statistics/{}.pickle'.format(crops_folder_name)

    info = get_info(dataset_path)

    with open(output_filename, 'wb') as file:
        pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset_info(filename: str) -> DatasetInfo:
    with open(filename, 'rb') as file:
        return pickle.load(file)


def create_crops(
        source_path: str,
        size_x: int,
        size_y: int,
        step_x: int,
        step_y: int,
        mask_converter: Callable[[np.ndarray], np.ndarray] = identity
):
    crops_path = '{}_crops'.format(source_path)
    clear_and_create(crops_path)

    args = get_data_paths(source_path)
    generator = dataset_generator(*args,
                                  size_x=size_x,
                                  size_y=size_y,
                                  step_x=step_x,
                                  step_y=step_y,
                                  mask_converter=mask_converter)

    nats = itertools.count(start=0, step=1)
    for n, (img, mask) in zip(nats, generator):
        cv2.imwrite('{}/{}_img.jpg'.format(crops_path, n), img)
        cv2.imwrite('{}/{}_mask.png'.format(crops_path, n), mask)


def write_dataset_part(
        orig_dataset_path: str,
        new_dataset_path: str,
        indices: List[int],
        converter: Callable[[np.ndarray], np.ndarray]
) -> None:
    orig_img_template = '%s/{}_img.jpg' % orig_dataset_path
    new_img_template = '%s/{}_img.jpg' % new_dataset_path
    orig_mask_template = '%s/{}_mask.png' % orig_dataset_path
    new_mask_template = '%s/{}_mask.png' % new_dataset_path

    for new_idx, orig_idx in enumerate(indices):
        img = cv2.imread(orig_img_template.format(orig_idx))
        mask = cv2.imread(orig_mask_template.format(orig_idx))
        bin_mask = converter(mask)

        cv2.imwrite(new_img_template.format(new_idx), img)
        cv2.imwrite(new_mask_template.format(new_idx), bin_mask)


INF = 1_000_000_000


def make_class_dataset(
        class_name: str,
        crops_folder_name: str,
        pure_percentage: float = 1,
        others_percentage: float = 1,
        max_size: int = INF,
        test_size: float = 0.2
) -> None:
    col = CLASS_TO_COL[class_name]
    info = load_dataset_info('statistics/{}.pickle'.format(crops_folder_name))

    mixed = list(info.get_mixed(col))
    pure = list(info.get_pure(col))
    others = list(info.get_others(col))

    mixed_len = len(mixed)
    pure_len = int(mixed_len * pure_percentage)
    others_len = int(mixed_len * others_percentage)

    result_indices = mixed[:mixed_len] + pure[:pure_len] + others[:others_len]
    shuffle(result_indices)
    result_indices = result_indices[:max_size]
    dataset_size = len(result_indices)

    test_indices = result_indices[:int(dataset_size * test_size)]
    train_indices = result_indices[int(dataset_size * test_size):]

    train_path = 'data/{}_train'.format(class_name)
    test_path = 'data/{}_test'.format(class_name)
    crops_path = 'data/{}'.format(crops_folder_name)

    clear_and_create(train_path)
    clear_and_create(test_path)

    converter = TO_BIN_CONVERTERS[class_name]

    write_dataset_part(crops_path, train_path, train_indices, converter)
    write_dataset_part(crops_path, test_path, test_indices, converter)


if __name__ == '__main__':
    # create_crops(
    #     source_path='data/water',
    #     size_x=224,
    #     size_y=224,
    #     step_x=112,
    #     step_y=112
    # )

    # calc_and_save_info('water_crops')

    # info = load_dataset_info('statistics/water_crops.pickle')
    # water_mixed = info.get_pure(CLASS_TO_COL['forest'])
    # for i in water_mixed:
    #     print(i)

    make_class_dataset(
        'water',
        'water_crops',
        pure_percentage=0.5,
        others_percentage=0.5,
    max_size=100
    )

    # for i in range(len(info.statistics)):
    #     cur_stat = info.statistics[i]
    #     print('{})'.format(i))
    #     for col in cur_stat:
    #         print('{}: {}'.format(col, cur_stat[col]))
