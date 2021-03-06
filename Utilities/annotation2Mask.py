import math

import numpy as np


def annotation2Mask(annotation, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if isinstance(annotation, float) and math.isnan(annotation):
        return np.zeros(shape)
    s = annotation.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def annotation2area(annotation):
    if isinstance(annotation, float) and math.isnan(annotation):
        return 0
    s = annotation.split()
    return np.asarray(s[1:][::2], dtype=int).sum()
