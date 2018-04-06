import sys
from cv2 import imread
from itertools import chain

import numpy as np

import cv2

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


def generate_224(img, size_x, size_y, step_x, step_y):
    height, width, _ = img.shape

    assert height >= step_x and height >= size_x
    assert width >= step_y and width >= size_y

    for x in range(0, height, step_x):
        for y in range(0, width, step_y):
            if x + size_x < height and y + size_y < width:
                yield img[x:x + size_x, y:y + size_y]


def get_data_generator(image_path, mask_path, size_x, size_y, step_x, step_y, mask_converter=convert_to_binary_water):
    img = imread(image_path)
    mask = imread(mask_path)
    assert img.shape == mask.shape

    x_generator = generate_224(img, size_x, size_y, step_x, step_y)
    bin_mask = mask_converter(mask)
    y_generator = generate_224(bin_mask, size_x, size_y, step_x, step_y)

    return zip(x_generator, y_generator)


def dataset_generator(
        *args,
        size_x,
        size_y,
        step_x,
        step_y,
        mask_converter=convert_to_binary_water
):
    generators = [
        get_data_generator(image_path, mask_path, size_x, size_y, step_x, step_y, mask_converter)
        for (image_path, mask_path) in args
    ]

    return chain(*generators)


def show(img, name="kek"):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def main(args):
    if len(args) != 6:
        sys.stderr.write(USAGE)
        return

    image_path, mask_path, size_x_str, size_y_str, step_x_str, step_y_str = args
    size_x, size_y, step_x, step_y = map(int, (size_x_str, size_y_str, step_x_str, step_y_str))

    for (img, mask) in get_data_generator(image_path, mask_path, size_x, size_y, step_x, step_y):
        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])
