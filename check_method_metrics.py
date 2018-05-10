import cv2

from mask_converters import *
from metrics import class_jacard_index

if __name__ == '__main__':
    methods = {'svm', 'unet'}
    img_name = '00.32953'
    mask = cv2.imread(f'tmp/{img_name}_mask.png')
    for method in methods:
        pred = cv2.imread(f'tmp/{img_name}_pred_{method}.png')

        print(f'{method}:')
        for t in TO_BIN_CONVERTERS:
            cur_conv = TO_BIN_CONVERTERS[t]
            print(f'\t{t}: {class_jacard_index(mask, pred, cur_conv)}')
