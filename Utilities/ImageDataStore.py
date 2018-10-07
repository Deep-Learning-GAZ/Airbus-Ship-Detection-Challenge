import os
from typing import Generator, List, Tuple

import numpy as np
from matplotlib.pyplot import imread


def imageDataStore(image_file_names: List[str], labels: List, batch_size: int) -> Generator[Tuple]:
    """
    Returns a generator, that iterates through the list of image file names, loads the image from the hard drive and
    returns a batch of it, along with the labels. If the labels are file names, thous are loaded as well.
    :param image_file_names: List of image file names
    :param labels: List of  image file names or some custom structure
    :param batch_size: Number of value pairs to return per iteration. (The last iteration can be smaller.)
    """
    assert len(image_file_names) == len(labels), "The length of image_file_names and labels muss be the same!"

    img_size, is_label_file = _getDataFormatInfo(image_file_names, labels)
    n_images = len(image_file_names)
    for batch_id in range(0, n_images, batch_size):
        n_yielded_images = batch_id * n_images
        end_of_batch = min(n_yielded_images + batch_size, len(image_file_names))
        current_batch_size = end_of_batch - n_yielded_images
        images = np.zeros((current_batch_size,) + img_size)
        if is_label_file:
            labels_binary = np.zeros((current_batch_size,) + img_size)
        else:
            labels_binary = []
        for image_id in range(n_yielded_images, end_of_batch):
            img = imread(image_file_names[image_id])
            images[image_id, :] = img
            label = labels[image_id]
            if is_label_file:
                labels_binary[image_id, :] = imread(label)
            else:
                labels_binary.append(label)
        yield (images, labels_binary)


def _getDataFormatInfo(image_file_names: List[str], labels: List[str]) -> Tuple[Tuple[int, int, int], bool]:
    test_img = imread(image_file_names[0])
    img_size = test_img.shape
    is_label_file = isinstance(labels[0], str) and os.path.exists(labels[0])
    return img_size, is_label_file
