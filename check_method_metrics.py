import cv2

from mask_converters import *
from metrics import class_jacard_index

if __name__ == '__main__':
    # methods = {'unet', 'svm', 'rtrees', 'mlp', 'knearest', 'boost'}
    methods = {'unet'}
    img_name = '00.32953'
    mask = cv2.imread('tmp/{}_mask.png'.format(img_name))
    for method in methods:
        pred = cv2.imread('tmp/{}_pred_{}.png'.format(img_name, method))

        print('{}:'.format(method))
        for t in TO_BIN_CONVERTERS:
            cur_conv = TO_BIN_CONVERTERS[t]
            score = class_jacard_index(mask, pred, cur_conv)
            print('\t{}: {}'.format(t, score if score != 1.0 else "not present"))
