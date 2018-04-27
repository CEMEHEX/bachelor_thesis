from typing import Tuple, Dict

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

COLOR_2_TYPE: Dict[ColorT, int] = \
    {TERRAIN_COL: 0,
     SNOW_COL: 1,
     SAND_COL: 2,
     FOREST_COL: 3,
     GRASS_COL: 4,
     ROADS_COL: 5,
     BUILDINGS_COL: 6,
     WATER_COL: 7,
     CLOUDS_COL: 8}

TYPE_2_COLOR: Dict[int, ColorT] = \
    {0: TERRAIN_COL,
     1: SNOW_COL,
     2: SAND_COL,
     3: FOREST_COL,
     4: GRASS_COL,
     5: ROADS_COL,
     6: BUILDINGS_COL,
     7: WATER_COL,
     8: CLOUDS_COL}
