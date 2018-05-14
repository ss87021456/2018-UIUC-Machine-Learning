"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = None
    data['label'] = None

    image_list = list()
    image_label = list()

    with open(data_txt_file, 'r') as data_txt:
        data_info = data_txt.read().splitlines()
        for image_info in data_info:
            image_path = image_data_path + image_info.split(',')[0]
            image_list.append(io.imread(image_path + '.jpg'))
            image_label.append(int(image_info.split(',')[1]))

    data['image'] = np.asarray(image_list)
    data['label'] = np.asarray(image_label)[:, np.newaxis]
    # print(data['image'].shape)
    # print(image_list[0].shape)
    # print(image_list[0])
    # print(image_label[0])
    # exit()
    return data
