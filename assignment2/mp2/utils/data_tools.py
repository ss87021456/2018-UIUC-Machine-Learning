"""Implements feature extraction and data processing helpers.
"""


import numpy as np


def preprocess_data(dataset,
                    feature_columns=[
                        'Id', 'BldgType', 'OverallQual'
                        'GrLivArea', 'GarageArea'
                    ],
                    squared_features=False,
                    ):
    """Processes the dataset into vector representation.

    When converting the BldgType to a vector, use one-hot encoding, the order
    has been provided in the one_hot_bldg_type helper function. Otherwise,
    the values in the column can be directly used.

    If squared_features is true, then the feature values should be
    element-wise squared.

    Args:
        dataset(dict): Dataset extracted from io_tools.read_dataset
        feature_columns(list): List of feature names.
        squred_features(bool): Whether to square the features.

    Returns:
        processed_datas(list): List of numpy arrays x,y.
            x is a numpy array, of dimension (N,K), N is the number of example
            in the dataset, and K is the len(feature_columns). Each row of x
            contains an example.
            y is a numpy array, of dimension (N,1) containing the SalePrice.
    """
    columns_to_id = {'Id': 0, 'BldgType': 1, 'OverallQual': 2,
                     'GrLivArea': 3, 'GarageArea': 4, 'SalePrice': 5}
    tmp_x = list()
    y = list()

    # append raw dataset
    for k in dataset:
        # skip first column 'id'
        tmp_x.append(dataset[k][1:-1])
        y.append(int(dataset[k][-1]))
    # print (tmp_x)
    # exit()
    x = list()
    cat_feature = list()
    for elements in tmp_x:
        tmp = list()
        for element in elements:
            try:
                # cast numerical feature into int
                if squared_features:
                    tmp.append(int(element) ** 2)
                else:
                    tmp.append(int(element))
            except ValueError:
                # deal with categorical feature
                element = one_hot_bldg_type(element)
                for ele in element:
                    tmp.append(ele)
                # cat_feature.append(element)
                pass
        x.append(tmp)
        del tmp

    del tmp_x
    # concate x follow by one-hot feature
    # for idx, feature in enumerate(cat_feature):
    # for element in feature:
    # x[idx].append(element)
    # print (x)
    x = np.asarray(x)
    # print (x.shape)
    # print (x[0])
    y = np.reshape(np.array(y), (len(y), 1))
    processed_dataset = [x, y]
    # pp.pprint(x)
    # exit()
    return processed_dataset


def one_hot_bldg_type(bldg_type):
    """Builds the one-hot encoding vector.

    Args:
        bldg_type(str): String indicating the building type.

    Returns:
        ret(list): A list representing the one-hot encoding vector.
            (e.g. for 1Fam building type, the returned list should be
            [1,0,0,0,0].
    """
    type_to_id = {'1Fam': 0,
                  '2FmCon': 1,
                  'Duplx': 2,
                  'TwnhsE': 3,
                  'TwnhsI': 4,
                  }
    ret = [0, 0, 0, 0, 0]
    for k in type_to_id:
        if k == bldg_type:
            ret[type_to_id[k]] = 1
    if sum(ret) == 0:
        print("there is an error!", bldg_type)

    return ret
