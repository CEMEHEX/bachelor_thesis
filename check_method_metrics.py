import cv2

from mask_converters import *
from metrics import class_jacard_index

if __name__ == '__main__':
    methods = {'unet', 'svm', 'rtrees', 'mlp', 'knearest', 'boost'}
    img_name = '00.32953'
    mask = cv2.imread(f'tmp/{img_name}_mask.png')
    for method in methods:
        pred = cv2.imread(f'tmp/{img_name}_pred_{method}.png')

        print(f'{method}:')
        for t in TO_BIN_CONVERTERS:
            cur_conv = TO_BIN_CONVERTERS[t]
            score = class_jacard_index(mask, pred, cur_conv)
            print(f'\t{t}: {score if score != 1.0 else "not present"}')
