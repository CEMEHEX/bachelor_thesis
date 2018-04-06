import time

import cv2
import numpy as np
from keras.optimizers import Adam

from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef


def prepare_model():
    model = ZF_UNET_224()
    model.load_weights("data/zf_unet_224.h5")  # optional
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    return model


def fit(model):
    # model.fit(...)
    pass


def main():
    start_time = time.time()

    model = prepare_model()

    check_on_simple_shapes(model)

    print('total time: ', time.time() - start_time)


if __name__ == '__main__':
    main()


def check_on_simple_shapes(model):
    img1 = cv2.imread("data/imgs/ex1.jpg")
    img2 = cv2.imread("data/imgs/ex2.jpg")
    imgs = np.array([img1, img2])

    res = model.predict(imgs)
    print(res.shape)

    for orig, res in zip(imgs, res):
        cv2.imshow("origin", orig)
        cv2.imshow("res", res)
        cv2.waitKey(0)
