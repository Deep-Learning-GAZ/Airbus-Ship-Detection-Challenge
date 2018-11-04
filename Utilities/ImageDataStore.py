import os
from typing import Callable, Generator, List, Tuple

import numpy as np
from matplotlib.pyplot import imread


def imageDataStore(image_file_names: List[str], labels: List, batch_size: int,
                   label_converter: Callable[[str], np.ndarray] = None,
                   image_converter: Callable[[str], np.ndarray] = None) -> Generator[
    Tuple[np.ndarray, np.ndarray], None, None]:
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

    img_size, label_size = _getDataFormatInfo(image_file_names, labels, label_converter, image_converter)
    n_images = len(image_file_names)
    while True:
        for n_yielded_images in range(0, n_images, batch_size):
            end_of_batch = min(n_yielded_images + batch_size, len(image_file_names))
            current_batch_size = end_of_batch - n_yielded_images
            images = np.zeros((current_batch_size,) + img_size)
            labels_binary = np.zeros((current_batch_size, label_size))
            for image_id in range(n_yielded_images, end_of_batch):
                image_id_batch = image_id - n_yielded_images
                img = imread(image_file_names[image_id])
                if image_converter is not None:
                    img = image_converter(img)
                images[image_id_batch, :] = img

                label = labels[image_id]
                if label_converter is not None:
                    label = label_converter(label)
                labels_binary[image_id_batch, :] = label
            yield (images, labels_binary)


def _getDataFormatInfo(image_file_names: List[str], labels: List,
                       label_converter: Callable[[str], np.ndarray] = None,
                       image_converter: Callable[[str], np.ndarray] = None) -> Tuple[Tuple[int, int, int], int]:
    test_img = image_converter(imread(image_file_names[0]))
    img_size = test_img.shape
    if label_converter is not None:
        label_shape = label_converter(labels[0]).shape
        assert (len(label_shape) == 1, "label_converter muss have 1-D return value!")
        label_size = label_shape[0]
    else:
        label_shape = labels[0].shape
        assert (len(label_shape) == 1, "If label_converter is not set, elements in labels muss be 1-D arrays!")
        label_size = label_shape[0]
    return img_size, label_size
