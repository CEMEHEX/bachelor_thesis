import time

import cv2
import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam

from split_generator import dataset_generator
from train_infinite_generator import batch_generator
from utils import get_data_paths
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef


def prepare_model() -> Model:
    model = ZF_UNET_224()
    model.load_weights("data/zf_unet_224.h5")  # optional
    # model.load_weights("data/zf_unet_224_water.h5")  # water weights
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    return model


def fit(model: Model):
    out_model_path = 'data/zf_unet_224_water.h5'
    epochs = 10
    batch_size = 15
    patience = 20

    args = get_data_paths("data/water")
    generator = dataset_generator(*args)

    def next_image():
        nonlocal generator
        try:
            res = next(generator)
        except StopIteration:
            generator = dataset_generator(*args)
            res = next(generator)

        return res

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('zf_unet_224_water_temp.h5', monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger('data/training.csv', append=True)
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size, next_image),
        epochs=epochs,
        steps_per_epoch=23,
        verbose=1,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('data/zf_unet_224_train_water.csv', index=False)
    print('Training is finished (weights zf_unet_224_water.h5 and log zf_unet_224_train_water.csv are generated )...')


def check_model(model: Model):
    # imgs = [cv2.imread('data/imgs/circle.png'), cv2.imread('data/imgs/ellipse.png')]
    cnt = 100
    imgs = []
    for i in range(cnt):
        print(f'data/splitted_water/ex{i + 1}.jpg')
        img = cv2.imread(f'data/splitted_water/ex{i + 1}.jpg')
        imgs.append(img)

    predicted_masks = model.predict(np.array(imgs))
    print(predicted_masks.shape)

    for orig, res in zip(imgs, predicted_masks):
        cv2.imshow("origin", orig)
        cv2.imshow("res", res)
        cv2.waitKey(0)


def main():
    start_time = time.time()

    model = prepare_model()
    fit(model)

    # check_model(model)

    print('total time: ', time.time() - start_time)


if __name__ == '__main__':
    main()
