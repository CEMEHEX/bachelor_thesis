import time
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.optimizers import Adam
import getopt
import sys
from batch_generator import DatasetSequence, preprocess_batch
from metrics import jacard_coef_loss
from utils import prepare_environment, view_images
from unet_model import get_unet


def prepare_model(weights: Optional[str] = None) -> Model:
    model = get_unet(batch_norm=True)
    if weights is not None:
        model.load_weights(weights)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=jacard_coef_loss, metrics=['accuracy'])

    return model


def fit(model: Model,
        out_model_path='weights/zf_unet_224_water.h5',
        train_path='data/water_train',
        test_path='data/water_train',
        epochs=10,
        batch_size=2,
        resume=False,
        last_epoch=-1):
    patience = 20

    train_generator = DatasetSequence(train_path, batch_size)
    test_generator = DatasetSequence(test_path, batch_size)

    steps_per_epoch = len(train_generator)

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='loss', patience=patience, verbose=1),
        ModelCheckpoint('weights/tmp/zf_unet_224_temp{epoch:02d}-{loss:.2f}.h5',
                        monitor='loss',
                        save_best_only=False,
                        verbose=1),
        CSVLogger('out/training.csv', append=resume),
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
        initial_epoch=last_epoch + 1 if resume else 0,
        verbose=1,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('out/zf_unet_224_train.csv', index=False)
    print('Training is finished (weights zf_unet_224.h5 and log zf_unet_224_train.csv are generated )...')


def check_model(model: Model,
                test_path='data/water_test',
                cnt=10):
    # imgs = [cv2.imread('data/imgs/ex1.png'), cv2.imread('data/imgs/ex2.png')]
    imgs = []
    for i in range(cnt):
        print(f'{test_path}/{i}_img.jpg')
        img = cv2.imread(f'{test_path}/{i}_img.jpg')
        imgs.append(img)

    imgs_preprocessed = np.array(imgs, dtype=np.float32)
    if K.image_dim_ordering() == 'th':
        imgs_preprocessed = imgs_preprocessed.transpose((0, 3, 1, 2))
    imgs_preprocessed = preprocess_batch(imgs_preprocessed)

    predicted_masks = model.predict(np.array(imgs_preprocessed))
    print(predicted_masks.shape)

    view_images([list(imgs), list(predicted_masks)], ['origin', 'res'])


def make_plots(source: str):
    df = pd.read_csv(source)

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(df['epoch'], df['acc'], label='train')
    plt.plot(df['epoch'], df['val_acc'], label='test')
    plt.legend(loc=0)
    plt.savefig('out/acc_plot.png')
    plt.gcf().clear()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(df['epoch'], df['loss'], label='train')
    plt.plot(df['epoch'], df['val_loss'], label='test')
    plt.legend(loc=0)
    plt.savefig('out/loss_plot.png')
    plt.gcf().clear()


def main():
    prepare_environment()
    start_time = time.time()
    target_class_name = 'water'

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            '',
            ['batch_size=', 'epochs=', 'weights=', 'train_data=', 'test_data=', 'cnt=', 'train', 'apply'])
    except Exception as e:
        print(e)
        sys.exit(2)

    opts = dict(opts)
    opts.setdefault('--batch_size', 2)
    opts.setdefault('--epochs', 50)
    opts.setdefault('--cnt', 10)

    train_mode, apply_mode, batch_size, epochs, cnt = False, False, 2, 50, 10
    weights_path, train_data_path, test_data_path = None, None, None
    for o in opts:
        if o == '--batch_size':
            batch_size = int(opts[o])
        elif o == '--epochs':
            epochs = int(opts[o])
        elif o == '--cnt':
            cnt = int(opts[o])
        elif o == '--weights':
            weights_path = opts[o]
        elif o == '--train_data':
            train_data_path = opts[o]
        elif o == '--test_data':
            test_data_path = opts[o]
        elif o == '--train':
            train_mode = True
        elif o == '--apply':
            apply_mode = True

    assert not (train_mode and apply_mode), "can't run in train and apply mode simultaneously"
    assert train_mode or weights_path is not None, "please specify weights_path"
    assert apply_mode or train_data_path is not None, "please specify train_data_path"
    assert test_data_path is not None, "please specify test data path"

    if train_mode:
        model = prepare_model('data/pretrained_weights.h5')  # pretrained
        fit(model, out_model_path=f'out/{target_class_name}.h5',
            train_path=train_data_path,
            test_path=test_data_path,
            epochs=epochs,
            batch_size=batch_size)
        make_plots('out/training.csv')
    else:
        model = prepare_model(weights_path)  # result weights
        check_model(model, test_data_path, cnt=cnt)

    print(f'total time: {(time.time() - start_time) / 3600.}h')


if __name__ == '__main__':
    main()
