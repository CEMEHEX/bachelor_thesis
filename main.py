import time
from typing import Optional

from keras import backend as K
import cv2
import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

from batch_generator import DatasetSequence, preprocess_batch
from metrics import jacard_coef_loss
from zf_unet_224_model import ZF_UNET_224


def prepare_model(weights: Optional[str] = None) -> Model:
    model = ZF_UNET_224()
    if weights is not None:
        model.load_weights(weights)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=jacard_coef_loss, metrics=['accuracy'])

    return model


def fit(model: Model, out_model_path='weights/zf_unet_224_water.h5'):
    epochs = 100
    batch_size = 2
    patience = 20

    train_generator = DatasetSequence('data/water_overfit_train', batch_size)
    test_generator = DatasetSequence('data/water_overfit_test', batch_size)

    steps_per_epoch = len(train_generator)

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='loss', patience=patience, verbose=1),
        # ModelCheckpoint('weights/zf_unet_224_water_temp{epoch:02d}-{loss:.2f}.h5',
        #                 monitor='loss',
        #                 save_best_only=False,
        #                 verbose=1),
        CSVLogger('data/training.csv', append=False),
        TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    batch_size=batch_size,
                    write_graph=True,
                    write_grads=False,
                    write_images=True,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None)

    ]

    print('Start training...')
    history = model.fit_generator(
        generator=train_generator,
        # validation_data=test_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('data/zf_unet_224_train_water.csv', index=False)
    print('Training is finished (weights zf_unet_224_water.h5 and log zf_unet_224_train_water.csv are generated )...')


def check_model(model: Model):
    # imgs = [cv2.imread('data/imgs/ex1.png'), cv2.imread('data/imgs/ex2.png')]
    imgs = []
    cnt = 50
    for i in range(cnt):
        print(f'data/water_test/{i}_img.jpg')
        img = cv2.imread(f'data/water_test/{i}_img.jpg')
        imgs.append(img)

    imgs_preprocessed = np.array(imgs, dtype=np.float32)
    if K.image_dim_ordering() == 'th':
        imgs_preprocessed = imgs_preprocessed.transpose((0, 3, 1, 2))
    imgs_preprocessed = preprocess_batch(imgs_preprocessed)

    predicted_masks = model.predict(np.array(imgs_preprocessed))
    print(predicted_masks.shape)

    for orig, res in zip(imgs, predicted_masks):
        cv2.imshow("origin", orig)
        cv2.imshow("res", res)
        cv2.waitKey(0)


def make_plots(source='data/zf_unet_224_train_water.csv'):
    df = pd.read_csv(source)

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(df['epoch'], df['acc'], label='train')
    # plt.plot(df['epoch'], df['val_acc'], label='test')
    plt.legend(loc=0)
    plt.savefig('data/acc_plot.png')
    plt.gcf().clear()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(df['epoch'], df['loss'], label='train')
    # plt.plot(df['epoch'], df['val_loss'], label='test')
    plt.legend(loc=0)
    plt.savefig('data/loss_plot.png')
    plt.gcf().clear()


def main():
    start_time = time.time()

    model = prepare_model('weights/zf_unet_224_water.h5')  # water weights
    # model = prepare_model('data/zf_unet_224.h5')  # pretrained

    # fit(model)
    # make_plots('data/training.csv')

    check_model(model)

    print(f'total time: {(time.time() - start_time) / 1000.0}h')


if __name__ == '__main__':
    main()
