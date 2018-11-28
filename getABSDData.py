import math
import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Utilities import annotation2Mask, annotation2area, imageDataStore, joinFolder


def getABSDData(batch_size: int, label_converter: Callable[[str], np.ndarray] = None, folder: str = 'data',
                reduced_size=None, remove_nan=True, area_limit=math.inf) \
        -> Tuple[imageDataStore, imageDataStore, imageDataStore]:
    """
    Creates 3 generators for train, dev and test sets.
    :param batch_size: Batch size.
    :param label_converter: Function. As input it gets the individual label element and the return value will be
    yielded by the generator. The return value should be a 1-D array, and have the same size for every input.
    :param folder: Location of the .csv files.
    :return: 3 generators for train, dev and test sets.
    """
    training_dfl, dev_df, test_df = getABSDDataFrames(folder, reduced_size, remove_nan, area_limit)

    def df2generator(df: pd.DataFrame) -> imageDataStore:
        image_file_names = df.ImageId.tolist()
        labels = df.EncodedPixels.tolist()
        return imageDataStore(image_file_names, labels, batch_size, label_converter)

    return df2generator(training_dfl), df2generator(dev_df), df2generator(test_df)


def getABSDDataMask(batch_size: int, label_converter: Callable[[np.ndarray], np.ndarray] = None, folder: str = 'data',
                    image_converter: Callable[[np.ndarray], np.ndarray] = None, reduced_size=None, remove_nan=True,
                    area_limit=math.inf) \
        -> Tuple[imageDataStore, imageDataStore, imageDataStore]:
    """
    Creates 3 generators for train, dev and test sets. The label is converted to mask.
    :param batch_size: Batch size.
    :param label_converter: (Optional) Function. As input it gets the mask, generated from the label element and the
    return value will be yielded by the generator. The return value should be a 1-D array, and have the same size for
    every input.
    :param folder: Location of the .csv files.
    :return: 3 generators for train, dev and test sets.
    """
    training_dfl, dev_df, test_df = getABSDDataFrames(folder, reduced_size, remove_nan, area_limit)

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

    def df2generator(df: pd.DataFrame) -> imageDataStore:
        groupped_df = df.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
        image_file_names = groupped_df.ImageId.tolist()
        labels = groupped_df.EncodedPixels.tolist()
        if label_converter is not None:
            converter = lambda x: label_converter(createUnitedMask(x)).flatten()
        else:
            converter = lambda x: createUnitedMask(x).flatten()
        return imageDataStore(image_file_names, labels, batch_size, converter, image_converter)

    return df2generator(training_dfl), df2generator(dev_df), df2generator(test_df)


def getABSDDataFrames(folder: str = 'data', reduced_size=None, remove_nan=True, area_limit=math.inf) -> Tuple[
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

    area_mask = [annotation2area(ann) > area_limit for ann in data.EncodedPixels]
    data = data[area_mask]

    train_image_names, test_image_names, dev_image_names = _shuffleImageNames(data, reduced_size)

    def selectImageFromData(image_names_to_select) -> pd.DataFrame:
        return data[data.ImageId.isin(image_names_to_select)]

    return selectImageFromData(train_image_names), selectImageFromData(dev_image_names), selectImageFromData(
        test_image_names)


def _shuffleImageNames(data, reduced_size) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_names = data.ImageId.unique()
    if reduced_size is not None:
        image_names = image_names[:reduced_size]
    train_dev_image_names, test_image_names, _, _ = train_test_split(image_names, range(0, len(image_names)),
                                                                     test_size=0.1, random_state=42)
    train_image_names, dev_image_names, _, _ = train_test_split(train_dev_image_names,
                                                                range(0, len(train_dev_image_names)), test_size=0.22,
                                                                random_state=42)
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
