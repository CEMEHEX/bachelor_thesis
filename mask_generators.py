import cv2
import numpy as np
from keras import Model
from typing import List, Type
from keras import backend as K

from batch_generator import preprocess_batch
from colors import TYPE_2_COLOR
from feature_extractor import chunk_descriptor
from main import prepare_model
from old_methods import OldModel, train_on_csv_data, SVM
from split_generator import generate_chunks_from_img


def compose_masks():
    pass

def stitch(img_parts: List[np.ndarray],
           orig_h: int,
           orig_w: int,
           x_cnt: int,
           y_cnt: int,
           dtype: Type[np.dtype]) -> np.ndarray:
    model_input_size = img_parts[0].shape[0]
    result = np.empty((orig_h, orig_w, 3), dtype=dtype)
    img_parts_iter = iter(img_parts)
    for y in range(1, y_cnt + 1):
        for x in range(1, x_cnt + 1):
            x2 = min(orig_w, x * model_input_size)
            y2 = min(orig_h, y * model_input_size)
            x1, y1 = x2 - model_input_size, y2 - model_input_size
            result[y1:y2, x1:x2] = next(img_parts_iter)
    return result


def unet_get_mask(model: Model,
                  img: np.ndarray,
                  model_input_size: int = 224) -> np.ndarray:
    height, width, _ = img.shape
    x_cnt = width // model_input_size if width % model_input_size == 0 else width // model_input_size + 1
    y_cnt = height // model_input_size if height % model_input_size == 0 else height // model_input_size + 1
    img_parts = []
    for y in range(1, y_cnt + 1):
        for x in range(1, x_cnt + 1):
            x2 = min(width, x * model_input_size)
            y2 = min(height, y * model_input_size)
            x1, y1 = x2 - model_input_size, y2 - model_input_size
            img_parts.append(img[y1:y2, x1:x2])

    imgs_preprocessed = np.array(img_parts, dtype=np.float32)
    if K.image_dim_ordering() == 'th':
        imgs_preprocessed = imgs_preprocessed.transpose((0, 3, 1, 2))
    imgs_preprocessed = preprocess_batch(imgs_preprocessed)

    mask_parts = model.predict(imgs_preprocessed)
    mask = stitch(mask_parts, height, width, x_cnt, y_cnt, np.float32)

    return mask


def old_model_get_mask(model: OldModel, img: np.ndarray, chunk_size: int = 4) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_chunks = generate_chunks_from_img(
        img,
        size_x=chunk_size,
        size_y=chunk_size,
        step_x=chunk_size,
        step_y=chunk_size)

    print('Extracting features...')
    features = np.array([chunk_descriptor(chunk) for chunk in img_chunks], dtype=np.float32)
    print(f'Done, features shape: {features.shape}')
    print('Predicting...')
    mask_types = iter(map(lambda t: int(t) if 0 <= t <= 9 else 9, model.predict(features)))
    print('Done')
    height, width, _ = img.shape

    print('Generating mask...')
    mask = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(height // chunk_size):
        for x in range(width // chunk_size):
            cur_type = mask_types.__next__()
            cur_x, cur_y = chunk_size * x, chunk_size * y
            mask[cur_y:cur_y + chunk_size, cur_x:cur_x + chunk_size] = TYPE_2_COLOR[cur_type]
    print('Done!')

    print('Applying median blur...')
    mask = cv2.medianBlur(mask, 9)
    print('Done!')

    return mask


def test_old():
    model = SVM()
    train_on_csv_data(model, 'out/water_small_features.csv')
    # model.save('out/svm.yaml')
    # model.load('out/svm.yaml')

    img_name = '00.32953'
    img = cv2.imread(f'tmp/{img_name}.jpg')
    mask = old_model_get_mask(model, img)

    # cv2.imshow('yeee', mask)
    cv2.imwrite(f'tmp/{img_name}_pred.png', mask)
    cv2.waitKey(0)


def test_unet():
    img_name = '00.32953'
    class_name = 'water'
    img = cv2.imread(f'tmp/{img_name}.jpg')

    model = prepare_model(
        weights_path=f'weights/{class_name}.h5',
        input_size=224)
    mask = unet_get_mask(model, img)
    print('Mask generated!')

    mask *= 255
    mask = np.array(mask, dtype=np.uint8)
    cv2.imwrite(f'tmp/00_{class_name}_mask.png', mask)

    # cv2.imshow('demo', img)
    # cv2.waitKey(0)
    # cv2.imshow('demo', mask)
    # cv2.waitKey(0)


if __name__ == '__main__':
    test_unet()
