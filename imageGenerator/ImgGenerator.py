import numpy as np


class DataGenerator:
    def __init__(self, img_list: np.ndarray, label_list: np.ndarray, ratio=0.8, shuffle=True):
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.n_class = len(set([label for label in label_list]))
        n_c_check = [0] * self.n_class
        self.train_indices = []
        self.test_indices = []
        for i, label in enumerate(label_list):
            if n_c_check[label] < int(int(len(label_list) * ratio)/self.n_class):
                self.train_indices.append(i)
                n_c_check[label] += 1
            else:
                self.test_indices.append(i)

        if shuffle:
            np.random.shuffle(self.train_indices)
            np.random.shuffle(self.test_indices)

        self.train_size = len(self.train_indices)
        self.test_size = len(self.test_indices)

        for i in self.train_indices:
            self.train_images.append(img_list[i])
            self.train_labels.append(label_list[i])
        for i in self.test_indices:
            self.test_images.append(img_list[i])
            self.test_labels.append(label_list[i])

        self.train_images = np.array(self.train_images) / 255.
        self.train_labels = np.array(self.train_labels)
        self.test_images = np.array(self.test_images) / 255.
        self.test_labels = np.array(self.test_labels)
        print(f'train, test set len : {len(self.train_images)}, {len(self.test_images)}')

    def train_generator(self):
        for i in range(len(self.train_images)):
            yield self.train_images[i], self.train_labels[i]

    def test_generator(self):
        for i in range(len(self.test_images)):
            yield self.test_images[i], self.test_labels[i]

    def get_shape(self):
        return self.train_images[0].shape
