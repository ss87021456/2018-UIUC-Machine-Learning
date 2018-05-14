from sklearn import multiclass, svm

import time


def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    '''
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    '''
    if mode == 'ovr':
        model = multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=12345))
        # print("start training")
        start = time.time()
        model.fit(X_train, y_train)
        total = time.time() - start
        # print("elapsed time {:3.2f}, start predicting".format(total))
        result = (model.predict(X_train), model.predict(X_test))
        return result
    elif mode == 'ovo':
        model = multiclass.OneVsOneClassifier(svm.LinearSVC(random_state=12345))
        # print("start training")
        start = time.time()
        model.fit(X_train, y_train)
        total = time.time() - start
        # print("elapsed time {:3.2f}, start predicting".format(total))
        # print("start predicting")
        result = (model.predict(X_train), model.predict(X_test))
        return result
    else:
        model = svm.LinearSVC(multi_class='crammer_singer',random_state=12345)
        # print("start training")
        start = time.time()
        model.fit(X_train, y_train)
        total = time.time() - start
        # print("elapsed time {:3.2f} start predicting".format(total))
        # print("start predicting")
        result = (model.predict(X_train), model.predict(X_test))
        return result
