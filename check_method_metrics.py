import os

import cv2
from typing import Set, Optional

import re

from mask_converters import *
from mask_generators import run_old_methods
from metrics import class_jacard_index


def calc_metrics(
        img_name: str,
        methods: Set[str]
) -> Dict[str, Dict[str, Optional[float]]]:
    mask = cv2.imread('comparing/{}_mask.png'.format(img_name))
    res = {}
    for method in methods:
        pred = cv2.imread('comparing/{}_pred_{}.png'.format(img_name, method))

        # print('{}:'.format(method))
        cur_scores = {}
        for t in TO_BIN_CONVERTERS:
            cur_conv = TO_BIN_CONVERTERS[t]
            score = class_jacard_index(mask, pred, cur_conv)
            cur_scores[t] = score if score != 1.0 else None
            # print('\t{}: {}'.format(t, score if score != 1.0 else "not present"))
        res[method] = cur_scores
    return res

def apply_to_each_img(fun, path: str = 'comparing'):
    for _, _, files in os.walk("comparing"):
        for filename in files:
            if 'mask' in filename or 'pred' in filename:
                continue
            img_name = re.sub('.jpg', '', filename)
            print('processing {}...'.format(img_name))
            fun(img_name)


if __name__ == '__main__':
    methods = {
        # 'unet',
        'svm',
        'rtrees',
        'mlp',
        'knearest',
        'boost'
    }

    def predict_old(img_name: str):
        run_old_methods(img_name, out_path='comparing')

    apply_to_each_img(fun=predict_old)
