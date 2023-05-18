import imageio.v3 as iio
import numpy as np
import label
import os


def read_image(img):
    return iio.imread(img, extension='.tif')


def imgs_to_ndarray(data: list) -> tuple:
    img_list = []
    label_list = []
    try:
        for img_name in data:
            img_arr = read_image(img_name)
            img_list.append(img_arr)
            label_list.append(label.labeling(img_name))
    except RuntimeError as e:
        print('Err while reading the images')
        print(e)
    return np.array(img_list), np.array(label_list)


def data_recur_search(path, cls=None):
    if cls is None:
        cls = [0, 1, 2, 3, 4]
    data_list = []

    if type(path) == str:
        for root, dirs, files in os.walk(path, topdown=False):
            label_num = root.split('/')[-1][0]
            if label_num.isnumeric() and (int(label_num)-1) in cls:
                for file in files:
                    if '.tif' in file:
                        data_list.append(f'{root}/{file}')
        return data_list
    else:
        for p in path:
            for root, dirs, files in os.walk(p, topdown=False):
                label_num = root.split('/')[-1][0]
                if label_num.isnumeric() and (int(label_num)-1) in cls:
                    for file in files:
                        if '.tif' in file:
                            data_list.append(f'{root}/{file}')
        return data_list


def subsampling(images, labels, size: int):
    selec_images = []
    selec_labels = []
    classes = [[] for _ in range(len(set(labels)))]
    for index, label in enumerate(labels):
        classes[label].append(index)
    for cls in classes:
        selec_indices = np.random.choice(len(cls), size, replace=False)
        for selec_index in selec_indices:
            selec_images.append(images[cls[selec_index]].copy())
            selec_labels.append(labels[cls[selec_index]].copy())
    selec_images = np.array(selec_images)
    selec_labels = np.array(selec_labels)
    return selec_images, selec_labels
