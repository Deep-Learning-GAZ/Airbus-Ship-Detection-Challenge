import os
import pandas as pd
from sklearn.model_selection import train_test_split


def getABSDData(folder='data'):
    training_annotation_file = os.path.join(folder, "train_ship_segmentations.csv")
    test_annotation_file = os.path.join(folder, "test_ship_segmentations.csv")
    training_data = pd.read_csv(training_annotation_file)
    test_data = pd.read_csv(test_annotation_file)
    data: pd.DataFrame = pd.concat([training_data, test_data])
    image_id_cn = 'ImageId'
    image_names = data[image_id_cn].unique()
    train_dev_image_names, test_image_names, _, _ = train_test_split(image_names, range(0, len(image_names)),
                                                                     test_size=0.1, random_state=42)
    train_image_names, dev_image_names, _, _ = train_test_split(train_dev_image_names,
                                                                range(0, len(train_dev_image_names)), test_size=0.22,
                                                                random_state=42)
    __selectImageFromData = lambda image_names_to_select: data[data[image_id_cn].isin(image_names_to_select)]
    return __selectImageFromData(train_image_names), __selectImageFromData(dev_image_names), __selectImageFromData(
        test_image_names)

if __name__ == '__main__':
    getABSDData()
