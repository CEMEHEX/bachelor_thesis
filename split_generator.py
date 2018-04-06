import sys
from cv2 import imread
from itertools import chain

import numpy as np

import cv2

from utils import get_data_paths

USAGE = "USAGE: python split_generator.py <image_path> <mask_path> <sizeX> <sizeY> <stepX> <stepY>"
WHITE_COL = [255, 255, 255]
BLACK_COL = [0, 0, 0]
# WATER_COL = [0, 128, 255]
WATER_COL = [255, 128, 0]


def convert_to_binary(mask, primary_color):
    res = np.zeros(mask.shape, dtype=np.uint8)
    res[np.where((mask == primary_color).all(axis=2))] = WHITE_COL
    return res


def convert_to_binary_water(mask):
    return convert_to_binary(mask, primary_color=WATER_COL)


def generate_224(img_path, size_x, size_y, step_x, step_y):
    print(img_path)
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    assert height >= step_x and height >= size_x
    assert width >= step_y and width >= size_y

    for x in range(0, height, step_x):
        for y in range(0, width, step_y):
            if x + size_x < height and y + size_y < width:
                yield img[x:x + size_x, y:y + size_y]


def data_generator(img_path, mask_path, size_x, size_y, step_x, step_y, mask_converter=convert_to_binary_water):
    x_generator = generate_224(img_path, size_x, size_y, step_x, step_y)
    y_generator = map(mask_converter, generate_224(mask_path, size_x, size_y, step_x, step_y))

    return x_generator, y_generator


def dataset_generator(
        *args,
        size_x=244,
        size_y=244,
        step_x=244,
        step_y=244,
        mask_converter=convert_to_binary_water
):
    generators = [
        data_generator(image_path, mask_path, size_x, size_y, step_x, step_y, mask_converter)
        for (image_path, mask_path) in args]

    x_gens, y_gens = zip(*generators)

    return chain(*x_gens), chain(*y_gens)


def dataset_gen_sample():
    args = [("/home/semyon/Programming/bachelor_thesis/terrain/unsorted/00.11884.jpg",
             "/home/semyon/Programming/bachelor_thesis/terrain/unsorted/00.11884_mask.png"),
            ("/home/semyon/Programming/bachelor_thesis/terrain/unsorted/01.30800.jpg",
             "/home/semyon/Programming/bachelor_thesis/terrain/unsorted/01.30800_mask.png"),
            ("/home/semyon/Programming/bachelor_thesis/terrain/unsorted/08.99471.jpg",
             "/home/semyon/Programming/bachelor_thesis/terrain/unsorted/08.99471_mask.png")]
    img_gen, mask_gen = dataset_generator(*args)

    for img, mask in zip(img_gen, mask_gen):
        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)


def dataset_from_dir_sample():
    args = get_data_paths("/home/semyon/Programming/Python/ZF_UNET_224/data/water")

    img_gen, mask_gen = dataset_generator(*args)

    cnt = 0
    for img, mask in zip(img_gen, mask_gen):
        cnt += 1
        print('%d)' % cnt)

        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)

    print('total count:', cnt)


def main(_):
    # if len(args) != 6:
    #     sys.stderr.write(USAGE)
    #     return
    #
    # image_path, mask_path, size_x_str, size_y_str, step_x_str, step_y_str = args
    # size_x, size_y, step_x, step_y = map(int, (size_x_str, size_y_str, step_x_str, step_y_str))
    #
    # for (img, mask) in data_generator(image_path, mask_path, size_x, size_y, step_x, step_y):
    #     cv2.imshow("img", img)
    #     cv2.imshow("mask", mask)
    #     cv2.waitKey(0)

    dataset_from_dir_sample()


if __name__ == '__main__':
    main(sys.argv[1:])
