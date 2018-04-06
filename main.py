import time

import cv2
import numpy as np
import pandas as pd

from keras.optimizers import Adam
from split_generator import dataset_generator
from train_infinite_generator import batch_generator
from utils import get_data_paths
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef


def prepare_model():
    model = ZF_UNET_224()
    model.load_weights("data/zf_unet_224.h5")  # optional
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    return model


def fit(model):
    out_model_path = 'data/zf_unet_224_water.h5'
    epochs = 200
    batch_size = 16

    args = get_data_paths("data/water")
    generator = dataset_generator(*args)

    def next_image():
        return generator.__next__()

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size, next_image),
        epochs=epochs,
        steps_per_epoch=200,
        verbose=2)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('data/zf_unet_224_train_water.csv', index=False)
    print('Training is finished (weights zf_unet_224_water.h5 and log zf_unet_224_train_water.csv are generated )...')


def main():
    start_time = time.time()

    model = prepare_model()
    fit(model)

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
