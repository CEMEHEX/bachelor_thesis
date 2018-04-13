import sys
from itertools import chain
from typing import Generator, Callable, Tuple, Iterator

import cv2
import numpy as np

WHITE_COL = 255
BLACK_COL = 0
WATER_COL = (255, 128, 0)

ColorT = Tuple[int, int, int]


def convert_to_binary(mask: np.ndarray, primary_color: ColorT) -> np.ndarray:
    res = np.zeros(mask.shape[0:2], dtype=np.uint8)
    res[np.where((mask == primary_color).all(axis=2))[0:2]] = WHITE_COL
    return res


def convert_to_binary_water(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=WATER_COL)


def generate_224(img_path: str,
                 size_x: int = 224,
                 size_y: int = 224,
                 step_x: int = 224,
                 step_y: int = 224) -> Generator[np.ndarray, None, None]:
    print(img_path)
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    assert height >= step_x and height >= size_x
    assert width >= step_y and width >= size_y

    for x in range(0, height, step_x):
        for y in range(0, width, step_y):
            if x + size_x < height and y + size_y < width:
                yield img[x:x + size_x, y:y + size_y]


def data_generator(img_path: str,
                   mask_path: str,
                   size_x: int,
                   size_y: int,
                   step_x: int,
                   step_y: int,
                   mask_converter: Callable[[np.ndarray], np.ndarray] = convert_to_binary_water
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    x_generator = generate_224(img_path, size_x, size_y, step_x, step_y)
    y_generator = map(mask_converter, generate_224(mask_path, size_x, size_y, step_x, step_y))

    return zip(x_generator, y_generator)


def dataset_generator(
        *args: Tuple[str, str],
        size_x: int = 224,
        size_y: int = 224,
        step_x: int = 224,
        step_y: int = 224,
        mask_converter: Callable[[np.ndarray], np.ndarray] = convert_to_binary_water
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    generators = [
        data_generator(image_path, mask_path, size_x, size_y, step_x, step_y, mask_converter)
        for (image_path, mask_path) in args]

    return chain(*generators)


def main(_):
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
