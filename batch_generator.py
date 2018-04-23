from random import shuffle
from typing import List

import cv2
import numpy as np
import os

import shutil
from keras import backend as K
from keras.utils import Sequence
from os.path import exists

from split_generator import dataset_generator
from utils import get_data_paths, files_cnt, have_diff_cols


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


def batch_generator(batch_size, next_image):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = next_image()
            image_list.append(img)
            mask_list.append([mask])

        image_list = np.array(image_list, dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)

        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0

        yield image_list, mask_list


def copy_from_tmp_folder(tmp_dir_path: str, dst_dir_path: str, indices: List[int]):
    i = 0
    for idx in indices:
        img = cv2.imread(f'{tmp_dir_path}/{idx}_img.jpg')
        mask = cv2.imread(f'{tmp_dir_path}/{idx}_mask.png', 0)

        cv2.imwrite(f'{dst_dir_path}/{i}_img.jpg', img)
        cv2.imwrite(f'{dst_dir_path}/{i}_mask.png', mask)

        i += 1


def prepare_data(source_path: str,
                 only_distinct: bool = True,
                 test_size: int = 0.2):
    tmp_dir_path = f'{source_path}_tmp'
    train_dir_path = f'{source_path}_train'
    test_dir_path = f'{source_path}_test'

    if exists(train_dir_path):
        shutil.rmtree(train_dir_path)
    if exists(test_dir_path):
        shutil.rmtree(test_dir_path)
    if exists(tmp_dir_path):
        shutil.rmtree(tmp_dir_path)

    os.makedirs(tmp_dir_path)
    os.makedirs(train_dir_path)
    os.makedirs(test_dir_path)

    args = get_data_paths(source_path)
    generator = dataset_generator(*args)

    # writing all images to tmp folder
    # TODO try to replace with zip infinite generator
    n = 0
    for img, mask in generator:
        if not only_distinct or have_diff_cols(mask):
            cv2.imwrite(f'{tmp_dir_path}/{n}_img.jpg', img)
            cv2.imwrite(f'{tmp_dir_path}/{n}_mask.png', mask)
            n += 1

    indices = list(range(n))
    shuffle(indices)

    test_cnt = int(n * test_size)
    train_indices = indices[test_cnt:]
    test_indices = indices[0:test_cnt]

    copy_from_tmp_folder(tmp_dir_path, train_dir_path, train_indices)
    copy_from_tmp_folder(tmp_dir_path, test_dir_path, test_indices)
    shutil.rmtree(tmp_dir_path)


class DatasetSequence(Sequence):
    def __init__(self, source_path: str, batch_size: int):
        self.source_path = source_path
        self.batch_size = batch_size
        self.cnt = files_cnt(source_path) // 2

    def __len__(self):
        return self.cnt // self.batch_size if self.cnt % self.batch_size == 0 else self.cnt // self.batch_size + 1

    def __getitem__(self, idx):
        i = idx * self.batch_size
        size = self.batch_size if i + self.batch_size < self.cnt else self.cnt - i

        image_list = map(lambda j: cv2.imread(f'{self.source_path}/{j}_img.jpg'), range(i, i + size))
        mask_list = map(lambda j: cv2.imread(f'{self.source_path}/{j}_mask.png', 0), range(i, i + size))
        mask_list = map(lambda x: x.reshape(224, 224, 1), mask_list)

        image_list = np.array(list(image_list), dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)
        mask_list = np.array(list(mask_list), dtype=np.float32)
        mask_list /= 255.0

        return image_list, mask_list

    def __iter__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item

    def on_epoch_end(self):
        pass


if __name__ == '__main__':
    seq = DatasetSequence('data/water_overfit_train', 2)
    for i, (img, mask) in zip(range(10), seq):
        print(f'{i})')
        print('\t', img.shape)
        print('\t', mask.shape)
    # wrapper = np.vectorize(lambda x: [x])
    # arr = np.array([[1, 2, 3], [4, 5, 6]])
    # print(arr.reshape((1, 2, 3)))