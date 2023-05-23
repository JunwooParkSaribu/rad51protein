import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from fileIO import imgs_to_ndarray, data_recur_search

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
    else:
        cur_path = '.'

    data_path = f'{cur_path}/data/selected_samples'
    model_path = f'{cur_path}/model/model5'
    batch_size = 8

    print(f'\nLoading the data...')
    data_list = data_recur_search(data_path, cls=[1])
    images, labels, names = imgs_to_ndarray(data_list)
    print(images.shape, labels.shape)

    Rad51_model = load_model(model_path, compile=True)
    Rad51_model.summary()
    y_pred = np.argmax((Rad51_model.predict(images / 255.)), axis=1)

    mismatchs = []
    for i in range(len(labels)):
        if labels[i] != y_pred[i]:
            mismatchs.append((images[i], labels[i], y_pred[i], names[i]))
    acc = [1 if x == 0 else 0 for x in (labels - y_pred)]
    acc = np.sum(acc)/len(acc)
    print(acc)
    """
    for mismatch in mismatchs:
        print(mismatch[1:3])
        print(mismatch[3])
        plt.figure()
        plt.imshow(mismatch[0])
        plt.show()
    """