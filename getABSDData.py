import os
from typing import Generator, Tuple, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from Utilities import imageDataStore
from Utilities import joinFolder
import numpy as np


def getABSDData(batch_size: int, label_converter: Callable[[str], np.ndarray] = None, folder: str = 'data') \
        -> Tuple[Generator[Tuple, None, None], Generator[Tuple, None, None], Generator[Tuple, None, None]]:
    """
    Creates 3 generators for train, dev and test sets.
    :param batch_size: Batch size.
    :param label_converter: Function. As input it gets the individual label element and the return value will be
    yielded by the generator. The return value should be a 1-D array, and have the same size for every input.
    :param folder: Location of the .csv files.
    :return: 3 generators for train, dev and test sets.
    """
    training_dfl, dev_df, test_df = getABSDDataFrames(folder)

    def df2generator(df: pd.DataFrame) -> Generator[Tuple, None, None]:
        image_file_names = df.ImageId.tolist()
        labels = df.EncodedPixels.tolist()
        return imageDataStore(image_file_names, labels, batch_size, label_converter)

    return df2generator(training_dfl), df2generator(dev_df), df2generator(test_df)


def getABSDDataFrames(folder: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits combines and splits the .csv files to train, dev and test sets.
    :param folder: The location of the .csv files.
    :return: DataFrames for train, dev and test sets.
    """
    training_annotation_file = os.path.join(folder, "train_ship_segmentations.csv")
    test_annotation_file = os.path.join(folder, "test_ship_segmentations.csv")
    training_data, test_data = _fixPaths(folder, training_annotation_file, test_annotation_file)

    data: pd.DataFrame = pd.concat([training_data, test_data])
    train_image_names, test_image_names, dev_image_names = _shuffleImageNames(data)

    def selectImageFromData(image_names_to_select) -> pd.DataFrame:
        return data[data.ImageId.isin(image_names_to_select)]

    return selectImageFromData(train_image_names), selectImageFromData(dev_image_names), selectImageFromData(
        test_image_names)


def _shuffleImageNames(data) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_names = data.ImageId.unique()
    train_dev_image_names, test_image_names, _, _ = train_test_split(image_names, range(0, len(image_names)),
                                                                     test_size=0.1, random_state=42)
    train_image_names, dev_image_names, _, _ = train_test_split(train_dev_image_names,
                                                                range(0, len(train_dev_image_names)), test_size=0.22,
                                                                random_state=42)
    return train_image_names, test_image_names, dev_image_names


def _fixPaths(folder: str, training_annotation_file: str, test_annotation_file: str)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_subfolder = os.path.join(folder, "train")
    test_subfolder = os.path.join(folder, "test")
    training_data = pd.read_csv(training_annotation_file)
    test_data = pd.read_csv(test_annotation_file)
    training_data.ImageId = joinFolder(train_subfolder, training_data.ImageId.tolist())
    test_data.ImageId = joinFolder(test_subfolder, test_data.ImageId.tolist())
    return training_data, test_data

