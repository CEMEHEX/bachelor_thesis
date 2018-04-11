import re
from os import listdir
from typing import Iterable, Tuple

from os.path import isfile, join


def get_name(filename: str,
             img_ext: str,
             mask_suffix: str,
             mask_ext: str) -> str:
    res = re.sub('\.' + img_ext, '', filename)
    res = re.sub(mask_suffix + '\.' + mask_ext, '', res)
    return res


def mask_name(filename: str, mask_suffix: str, mask_ext: str) -> str:
    return '%s%s.%s' % (filename, mask_suffix, mask_ext)


def origin_name(filename: str, img_ext: str) -> str:
    return '%s.%s' % (filename, img_ext)


def get_result(dir_path: str,
               filename: str,
               img_ext: str,
               mask_suffix: str,
               mask_ext: str) -> (str, str):
    origin_path = join(dir_path, origin_name(filename, img_ext))
    mask_path = join(dir_path, mask_name(filename, mask_suffix, mask_ext))
    return origin_path, mask_path


def get_data_paths(
        dir_path: str,
        img_ext: str = "jpg",
        mask_suffix: str = "_mask",
        mask_ext: str = "png") -> Iterable[Tuple[str, str]]:
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    files = map(lambda f: get_name(f, img_ext, mask_suffix, mask_ext), files)
    files = list(set(files))

    return list(map(lambda f: get_result(dir_path, f, img_ext, mask_suffix, mask_ext), files))


def main():
    res = get_data_paths("data/water")
    for r in res:
        print(r)


if __name__ == '__main__':
    main()
