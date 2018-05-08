import cv2
import numpy as np
from keras import Model

from colors import TYPE_2_COLOR
from feature_extractor import chunk_descriptor
from old_methods import OldModel, train_on_csv_data, SVM
from split_generator import generate_chunks_from_img

def unet_get_mask(model: Model, img: np.ndarray, model_input_size = 224):
    pass


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

if __name__ == '__main__':
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