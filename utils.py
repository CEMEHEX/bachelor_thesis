import re
from os import listdir, makedirs
from os.path import isfile, join
from random import shuffle
from typing import Iterable, Tuple, List
from os.path import exists
import numpy as np

import cv2


def files_cnt(dir_name: str):
    return len([name for name in listdir(f'./{dir_name}')])


def get_name(filename: str,
             img_ext: str = 'jpg',
             mask_suffix: str = '_mask',
             mask_ext: str = 'png') -> str:
    res = re.sub('\.' + img_ext, '', filename)
    res = re.sub(mask_suffix + '\.' + mask_ext, '', res)
    return res


def mask_name(filename: str, mask_suffix: str, mask_ext: str) -> str:
    return f'{filename}{mask_suffix}.{mask_ext}'


def origin_name(filename: str, img_ext: str) -> str:
    return f'{filename}.{img_ext}'


def get_result(dir_path: str,
               filename: str,
               img_ext: str,
               mask_suffix: str,
               mask_ext: str) -> (str, str):
    origin_path = join(dir_path, origin_name(filename, img_ext))
    mask_path = join(dir_path, mask_name(filename, mask_suffix, mask_ext))
    return origin_path, mask_path


# Generates (img_path, mask_path) pairs list from specified folder
def get_data_paths(
        dir_path: str,
        img_ext: str = "jpg",
        mask_suffix: str = "_mask",
        mask_ext: str = "png") -> Iterable[Tuple[str, str]]:
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    files = map(lambda f: get_name(f, img_ext, mask_suffix, mask_ext), files)
    files = list(set(files))

    return list(map(lambda f: get_result(dir_path, f, img_ext, mask_suffix, mask_ext), files))


def have_diff_cols(img) -> bool:
    height, width = img.shape
    return 0 < cv2.countNonZero(img) < height * width


def create_if_not_exists(dirs_path):
    if not exists(dirs_path):
        makedirs(dirs_path)
        print(dirs_path, 'has been created!')


def shuffle_csv(path_to_csv: str) -> None:
    fid = open(path_to_csv, "r")
    li = fid.readlines()
    fid.close()

    shuffle(li)

    fid = open("shuffled_example.txt", "w")
    fid.writelines(li)
    fid.close()



def prepare_environment():
    create_if_not_exists('data')
    create_if_not_exists('weights')
    create_if_not_exists('weights/tmp')
    create_if_not_exists('out')
    create_if_not_exists('results')

    create_if_not_exists('statistics')


def view_images(imgs: List[List[np.ndarray]],
                win_names: List[str]):
    LEFT = 37
    RIGHT = 39
    idx = 0
    c = 0
    while c != 27:  # escape
        for img_list, win_name in zip(imgs, win_names):
            cv2.imshow(win_name, img_list[idx])
        c = cv2.waitKeyEx(0)

        if c == LEFT:
            idx = (idx - 1) % len(imgs[0])
        elif c == RIGHT:
            idx = (idx + 1) % len(imgs[0])


def main():
    res = get_data_paths("data/water")
    for r in res:
        print(r)


if __name__ == '__main__':
    main()
