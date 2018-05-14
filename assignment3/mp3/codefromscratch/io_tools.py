"""Input and output helpers to load in data.
"""
import numpy as np


def read_dataset(path_to_dataset_folder, index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing
        samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature
                                of each sample

        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...]
                             where yi is +1/-1, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')

    label = list()
    feature = list()

    with open(path_to_dataset_folder+'/'+index_filename, 'r') as file_idx:
        dataset = file_idx.read().splitlines()  # split according to new_line

        for info in dataset:
            feature_file = info.split()[1]
            with open(path_to_dataset_folder+'/'+feature_file, 'r') as data:
                data = data.read()

                # process feature and label into correct dtype
                label.append(int(info.split()[0]))
                feature.append([float(i) for i in data.split()])

    feature = np.asarray(feature)
    T = np.asarray(label)
    A = np.concatenate((np.ones((feature.shape[0], 1)), feature), axis=1)

    return A, T
