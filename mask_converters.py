import numpy as np

from colors import *

WHITE_COL = 255
BLACK_COL = 0


def convert_to_binary(mask: np.ndarray, primary_color: ColorT) -> np.ndarray:
    res = np.zeros(mask.shape[0:2], dtype=np.uint8)
    res[np.where((mask == primary_color).all(axis=2))[0:2]] = WHITE_COL
    return res


def convert_to_binary_terrain(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=TERRAIN_COL)


def convert_to_binary_snow(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=SNOW_COL)


def convert_to_binary_sand(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=SAND_COL)


def convert_to_binary_forest(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=FOREST_COL)


def convert_to_binary_grass(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=GRASS_COL)


def convert_to_binary_roads(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=ROADS_COL)


def convert_to_binary_buildings(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=BUILDINGS_COL)


def convert_to_binary_water(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=WATER_COL)


def convert_to_binary_clouds(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=CLOUDS_COL)


def convert_from_binary(bin_mask: np.ndarray, primary_color: ColorT) -> np.ndarray:
    height, width = bin_mask.shape
    res = np.zeros((height, width, 3), dtype=np.uint8)
    res[np.where(bin_mask == WHITE_COL)] = primary_color
    return res


def convert_from_binary_terrain(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=TERRAIN_COL)


def convert_from_binary_snow(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=SNOW_COL)


def convert_from_binary_sand(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=SAND_COL)


def convert_from_binary_forest(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=FOREST_COL)


def convert_from_binary_grass(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=GRASS_COL)


def convert_from_binary_roads(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=ROADS_COL)


def convert_from_binary_buildings(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=BUILDINGS_COL)


def convert_from_binary_water(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=WATER_COL)


def convert_from_binary_clouds(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=CLOUDS_COL)


# if __name__ == '__main__':
#     import cv2
#
#     bin_mask = cv2.imread('tmp/full_size/00_forest_mask.png', 0)
#     mask = convert_from_binary_forest(bin_mask)
#     cv2.imshow('aa', mask)
#     cv2.waitKey(0)
