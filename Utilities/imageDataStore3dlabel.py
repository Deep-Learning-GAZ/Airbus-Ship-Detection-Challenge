import os
from typing import Callable, Generator, List, Tuple

import numpy as np
from keras.utils import Sequence
from matplotlib.pyplot import imread
from keras.utils import to_categorical


class imageDataStore3dlabel(Sequence):
    def __init__(self, image_file_names: List[str], labels: List, batch_size: int,
                 label_converter: Callable[[str], np.ndarray] = None,
                 image_converter: Callable[[str], np.ndarray] = None):
        """
        Returns a generator, that iterates infinitely through the list of image file names, loads the image from the hard
        drive and returns a batch of it, along with the labels. If len(image_file_names) % batch_size != 0, then last batch
        in the epoch is shorten, than the batch size.
        :param image_file_names: List of image file names.
        :param labels: List of same sized 1-D arrays, or some custom structure accepted by label_converter.
        :param batch_size: Number of value pairs to return per iteration. (The last iteration can be smaller.)
        :param label_converter: (Optional) Function. As input it gets the individual label element and the return value will
        be yielded by the generator. The return value should be a 1-D array, and have the same size for every input.
        """
        assert (len(image_file_names) == len(labels), "The length of image_file_names and labels muss be the same!")

        self.image_file_names = image_file_names
        self.labels = labels
        self.batch_size = batch_size
        self.label_converter = label_converter
        self.image_converter = image_converter
        self.n_images = len(image_file_names)

        self.img_size, self.label_size = self._getDataFormatInfo()

    def __len__(self):
        return int(np.ceil(self.n_images / float(self.batch_size)))

    def __getitem__(self, idx):
        n_yielded_images = idx * self.batch_size
        end_of_batch = min(n_yielded_images + self.batch_size, len(self.image_file_names))
        current_batch_size = end_of_batch - n_yielded_images
        images = np.zeros((current_batch_size,) + self.img_size)
        labels_binary = np.zeros((current_batch_size,) + self.label_size)
        for image_id in range(n_yielded_images, end_of_batch):
            image_id_batch = image_id - n_yielded_images
            images[image_id_batch, :] = self._getImage(image_id)
            labels_binary[image_id_batch, :] = self._getLabel(image_id)
        return images, labels_binary

    def _getLabel(self, image_id):
        label = self.labels[image_id]
        if self.label_converter is not None:
            label = self.label_converter(label)
        label = to_categorical(label, num_classes=2)
        return label

    def _getImage(self, image_id):
        img = imread(self.image_file_names[image_id])
        if self.image_converter is not None:
            img = self.image_converter(img)
        return img

    def _getDataFormatInfo(self) -> Tuple[Tuple[int, int, int], int]:
        test_img = self._getImage(0)
        img_size = test_img.shape
        label = self._getLabel(0)
        label_shape = label.shape
        return img_size, label_shape
