import shutil
from typing import Callable, Set, List, Dict

from colors import ColorT
from mask_converters import identity
import itertools

import cv2
import numpy as np
from os.path import exists
import os

from split_generator import dataset_generator
from utils import get_data_paths

# The n-th element of the list represents a dictionary that
# for each color stores the percentage of this color in the n-th mask
DatasetInfoT = List[Dict[ColorT, float]]



def create_crops(
        source_path: str,
        size_x: int,
        size_y: int,
        step_x: int,
        step_y: int,
        mask_converter: Callable[[np.ndarray], np.ndarray] = identity
):
    crops_path = '{}_crops'.format(source_path)

    if exists(crops_path):
        shutil.rmtree(crops_path)

    os.makedirs(crops_path)

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


# def calc_crops_stats

if __name__ == '__main__':
    classes = ['water', 'forest', 'buildings', 'roads']
    create_crops(
        source_path='data/water',
        size_x=224,
        size_y=224,
        step_x=112,
        step_y=112
    )
