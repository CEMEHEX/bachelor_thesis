from typing import Tuple

import numpy as np

WHITE_COL = 255
BLACK_COL = 0

ColorT = Tuple[int, int, int]

TERRAIN_COL = (0, 51, 102)
SNOW_COL = (255, 255, 204)
SAND_COL = (51, 255, 255)
FOREST_COL = (0, 102, 0)
GRASS_COL = (51, 255, 51)

ROADS_COL = (160, 160, 160)

BUILDINGS_COL = (96, 96, 96)

WATER_COL = (255, 128, 0)

CLOUDS_COL = (224, 224, 224)


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
