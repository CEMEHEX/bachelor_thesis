import time

import cv2
import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from batch_generator import DatasetSequence
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss


def prepare_model() -> Model:
    model = ZF_UNET_224()
    # model.load_weights("data/zf_unet_224.h5")  # optional
    model.load_weights("weights/zf_unet_224_water_temp01--0.47.h5")  # water weights
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=['accuracy'])

    return model


def fit(model: Model):
    out_model_path = 'weights/zf_unet_224_water.h5'
    epochs = 20
    batch_size = 2
    patience = 20

    train_generator = DatasetSequence('data/water_train', batch_size)
    test_generator = DatasetSequence('data/water_test', batch_size)

    steps_per_epoch = len(train_generator)

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='loss', patience=patience, verbose=1),
        ModelCheckpoint('weights/zf_unet_224_water_temp{epoch:02d}-{loss:.2f}.h5',
                        monitor='loss',
                        save_best_only=False,
                        verbose=1),
        CSVLogger('data/training_batch1.csv', append=True),
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
        validation_data=test_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
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

def make_plots(source='data/zf_unet_224_train_water.csv'):
    df = pd.read_csv(source)

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(df['epoch'], df['acc'], label='train')
    plt.plot(df['epoch'], df['val_acc'], label='test')
    plt.legend(loc=0)
    plt.savefig('data/acc_plot.png')
    plt.gcf().clear()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(df['epoch'], df['loss'], label='train')
    plt.plot(df['epoch'], df['val_loss'], label='test')
    plt.legend(loc=0)
    plt.savefig('data/loss_plot.png')
    plt.gcf().clear()


def main():
    start_time = time.time()

    # model = prepare_model()
    # fit(model)

    # check_model(model)

    make_plots()
    print(f'total time: {(time.time() - start_time) / 1000.0}h')


if __name__ == '__main__':
    main()
