from matplotlib.pyplot import imread
import os
import numpy as np
from typing import List


def imageDataStore(image_file_names: List[str], labels: List, batch_size: int):
    assert len(image_file_names) == len(labels)
    img_size, is_label_file = getDataFormatInfo(image_file_names, labels)
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


def getDataFormatInfo(image_file_names, labels):
    test_img = imread(image_file_names[0])
    img_size = test_img.shape
    is_label_file = isinstance(labels[0], str) and os.path.exists(labels[0])
    return img_size, is_label_file
