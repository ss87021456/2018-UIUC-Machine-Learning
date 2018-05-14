import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        #W = np.zeros((K, d))
        W = np.ones((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)
            if i % 10 == 0:
                loss = self.loss_student(W, X_intercept, y)
                #print(loss)
                print("step {:3d} / {:3d} loss:{:3.2f}".format(i, n_iter, loss[0][0]))

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        models = dict.fromkeys(self.labels, 0)
        
        for label in self.labels:
            temp_y = (y == label).astype(int) # generate data
            model = svm.LinearSVC(random_state=12345).fit(X, temp_y)
            models[label] = model

        return models

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        def combinations_2(num):
            tmp = list()
            num = len(num)
            for i in range(num):
                for j in range(i+1, num, 1):
                    tmp.append((i,j))
            return tmp

        dataset_X = dict.fromkeys(self.labels, 0)
        dataset_y = dict.fromkeys(self.labels, 0)
        for label in self.labels: # create separate dataset
            dataset_X[label] = X[y == label]
            dataset_y[label] = y[y == label]

        label_pairs = [c for c in combinations_2(self.labels)]
        models = dict.fromkeys(label_pairs, 0)
        for key in models:
            first, second  = key
            tmp_X = np.concatenate((dataset_X[first], dataset_X[second]), axis=0)
            tmp_y = np.concatenate((dataset_y[first], dataset_y[second]), axis=0)
            tmp_y = (tmp_y == second).astype(int) # convert to (first 0, second 1)
            #print(tmp_y[0],tmp_y[-1])
            #exit()
            model = svm.LinearSVC(random_state=12345).fit(tmp_X, tmp_y)
            models[key] = model

        return models


    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = list()
        for query in X:
            query = query.reshape(1, -1)
            single_score = list()
            for label in self.labels: # predict for each part
                single_score.append(self.binary_svm[label].decision_function(query)[0])
            single_score = np.array(single_score)
            scores.append(single_score)
        scores = np.array(scores)

        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = list()
        for query in X:
            query = query.reshape(1, -1)
            single_vote = np.zeros(len(self.labels))
            for key in self.binary_svm:
                first, second = key
                predict = self.binary_svm[key].predict(query)
                if predict == 0:
                    single_vote[first] += 1
                elif predict == 1:
                    single_vote[second] += 1
            scores.append(single_vote)
        scores = np.array(scores)

        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        regularization = 0.5 * np.sum(np.multiply(W,W))
        
        penalty = 0
        for x_i, y_i in zip(X, y):
            x = x_i.reshape(1, -1)
            argmax = -99999999
            for i in range(len(W)):
                term = 0
                if i == y_i:
                    term = x.dot(W[y_i][:, np.newaxis])
                else:
                    term = 1 + x.dot(W[i][:, np.newaxis])
                
                if term > argmax:
                    argmax = term
            penalty += argmax - x.dot(W[y_i][:, np.newaxis])

        total_loss = regularization + penalty
        #print(total_loss)
        return total_loss
        

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        def one_hot(x):
            n_values = len(np.unique(y))
            return np.eye(n_values)[x]

        regular_grad = W
        # penalty_grad = np.zeros(W.shape)
        max_idx = one_hot(np.argmax(((1 + X.dot(W.T)) - one_hot(y)), axis=1)).T # (10, 5000)
        partial_first = max_idx.dot(X) # (10, 785)
        partial_second = (one_hot(y).T).dot(X) # (10, 785)
        penalty_grad = partial_first - partial_second

        #print(max_idx[10])
        #exit()
        '''
        for x_i, y_i in zip(X, y):
            x = x_i.reshape(1, -1)
            for i in range(len(W)):
                correct_term = x.dot(W[y_i][:, np.newaxis])
                max_term = correct_term
                max_idx = y_i
                for j in range(len(W)):
                    if j != y_i:
                        wrong_term = 1 + x.dot(W[j][:, np.newaxis])
                        if wrong_term > max_term:
                            max_term = wrong_term
                            max_idx = j

                if max_idx != y_i and i == y_i:
                    penalty_grad[y_i] += -x.flatten()
                elif max_idx == i and i != y_i:
                    penalty_grad[i] += x.flatten()
        '''
        grad = regular_grad + C * penalty_grad

        return grad



