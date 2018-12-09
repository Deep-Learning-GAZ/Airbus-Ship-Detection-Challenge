import math
import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Utilities import annotation2Mask, annotation2area, joinFolder
from Utilities.imageDataStore3dlabel import imageDataStore3dlabel


def getABSDData(batch_size: int, label_converter: Callable[[str], np.ndarray] = None, folder: str = 'data',
                reduced_size=None, remove_nan=True) \
        -> Tuple[imageDataStore3dlabel, imageDataStore3dlabel, imageDataStore3dlabel]:
    """
    Creates 3 generators for train, dev and test sets.
    :param batch_size: Batch size.
    :param label_converter: Function. As input it gets the individual label element and the return value will be
    yielded by the generator. The return value should be a 1-D array, and have the same size for every input.
    :param folder: Location of the .csv files.
    :return: 3 generators for train, dev and test sets.
    """
    training_dfl, dev_df, test_df = getABSDDataFrames(folder, reduced_size, remove_nan)

    def df2generator(df: pd.DataFrame) -> imageDataStore3dlabel:
        image_file_names = df.ImageId.tolist()
        labels = df.EncodedPixels.tolist()
        return imageDataStore3dlabel(image_file_names, labels, batch_size, label_converter)

    return df2generator(training_dfl), df2generator(dev_df), df2generator(test_df)


def getABSDDataMask(batch_size: int, label_converter: Callable[[np.ndarray], np.ndarray] = None, folder: str = 'data',
                    image_converter: Callable[[np.ndarray], np.ndarray] = None, reduced_size=None, remove_nan=True) \
        -> Tuple[imageDataStore3dlabel, imageDataStore3dlabel, imageDataStore3dlabel]:
    """
    Creates 3 generators for train, dev and test sets. The label is converted to mask.
    :param batch_size: Batch size.
    :param label_converter: (Optional) Function. As input it gets the mask, generated from the label element and the
    return value will be yielded by the generator. The return value should be a 3-D array, and have the same size for
    every input.
    :param folder: Location of the .csv files.
    :return: 3 generators for train, dev and test sets.
    """
    training_dfl, dev_df, test_df = getABSDDataFrames(folder, reduced_size, remove_nan)

    def createUnitedMask(mask_str_arr: List[List[str]]) -> np.ndarray:
        assert (len(mask_str_arr) > 0)
        united_mask = None
        for mask_str in mask_str_arr:
            mask = annotation2Mask(mask_str)
            if united_mask is None:
                united_mask = mask
            else:
                united_mask += mask
        return united_mask

    def df2generator(df: pd.DataFrame) -> imageDataStore3dlabel:
        groupped_df = df.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
        image_file_names = groupped_df.ImageId.tolist()
        labels = groupped_df.EncodedPixels.tolist()
        if label_converter is not None:
            converter = lambda x: label_converter(createUnitedMask(x))
        else:
            converter = lambda x: createUnitedMask(x)
        return imageDataStore3dlabel(image_file_names, labels, batch_size, converter, image_converter)

    return df2generator(training_dfl), df2generator(dev_df), df2generator(test_df)


def getABSDDataFrames(folder: str = 'data', reduced_size=None, remove_nan=True) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits combines and splits the .csv files to train, dev and test sets.
    :param folder: The location of the .csv files.
    :return: DataFrames for train, dev and test sets.
    """
    training_annotation_file = os.path.join(folder, "train_ship_segmentations.csv")
    test_annotation_file = os.path.join(folder, "test_ship_segmentations.csv")
    training_data, test_data = _fixPaths(folder, training_annotation_file, test_annotation_file)

    data = pd.concat([training_data, test_data])
    if remove_nan:
        data = data.dropna()

    train_image_names, test_image_names, dev_image_names = _shuffleImageNames(data, reduced_size)

    def selectImageFromData(image_names_to_select) -> pd.DataFrame:
        return data[data.ImageId.isin(image_names_to_select)]

    return selectImageFromData(train_image_names), selectImageFromData(dev_image_names), selectImageFromData(
        test_image_names)


def _shuffleImageNames(data, reduced_size) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data['Area'] = [annotation2area(ann) for ann in data.EncodedPixels]

    # image_names = data.ImageId.unique()
    data = data.groupby('ImageId')['Area'].sum()
    data = shuffle(data, random_state=42)
    n_image = len(data)
    border_train = int(n_image * 0.7)
    border_dev = int(n_image * 0.9)
    train_image_names = data[:border_train]
    dev_image_names = data[border_train:border_dev]
    test_image_names = data[border_dev:]

    if reduced_size is not None:
        border_train = int(reduced_size * 0.7)
        border_dev = int(reduced_size * 0.2)
        border_test = reduced_size - border_train - border_dev
        train_image_names.sort_values(ascending=False, inplace=True)
        dev_image_names.sort_values(ascending=False, inplace=True)
        test_image_names.sort_values(ascending=False, inplace=True)
        train_image_names = train_image_names.index[:border_train].tolist()
        dev_image_names = dev_image_names.index[:border_dev].tolist()
        test_image_names = test_image_names.index[:border_test].tolist()
    else:
        train_image_names = train_image_names.index.tolist()
        dev_image_names = dev_image_names.index.tolist()
        test_image_names = test_image_names.index.tolist()
    return train_image_names, test_image_names, dev_image_names


def _fixPaths(folder: str, training_annotation_file: str, test_annotation_file: str) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_subfolder = os.path.join(folder, "train")
    test_subfolder = os.path.join(folder, "test")
    training_data = pd.read_csv(training_annotation_file)
    test_data = pd.read_csv(test_annotation_file)
    training_data.ImageId = joinFolder(train_subfolder, training_data.ImageId.tolist())
    test_data.ImageId = joinFolder(test_subfolder, test_data.ImageId.tolist())
    return training_data, test_data
