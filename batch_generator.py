import cv2
import numpy as np
import os

import shutil
from keras import backend as K
from keras.utils import Sequence

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


class DatasetSequence(Sequence):
    def __init__(self, dir_path: str, batch_size: int, update=True, only_distinct=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.output_dir_path = f'{dir_path}_dataset'

        if os.path.exists(self.output_dir_path):
            if not update:
                self.cnt = files_cnt(self.output_dir_path) // 2
                return
            else:
                shutil.rmtree(self.output_dir_path)
        os.makedirs(self.output_dir_path)

        args = get_data_paths(dir_path)
        generator = dataset_generator(*args)

        i = 0
        for img, mask in generator:
            if not only_distinct or have_diff_cols(mask):
                cv2.imwrite(f'{self.output_dir_path}/{i}_img.jpg', img)
                cv2.imwrite(f'{self.output_dir_path}/{i}_mask.png', mask)
                i += 1

        self.cnt = i

    def __len__(self):
        return self.cnt // self.batch_size if self.cnt % self.batch_size == 0 else self.cnt // self.batch_size + 1

    def __getitem__(self, idx):
        i = idx * self.batch_size
        size = self.batch_size if i + self.batch_size < self.cnt else self.cnt - i

        image_list = map(lambda j: cv2.imread(f'{self.output_dir_path}/{j}_img.jpg'), range(i, i + size))
        mask_list = map(lambda j: [cv2.imread(f'{self.output_dir_path}/{j}_mask.png', 0)], range(i, i + size))

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
