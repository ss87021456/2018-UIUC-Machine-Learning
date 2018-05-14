"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

label_dict = None

class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = np.random.rand(n_components, n_dims)
        # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        self._pi = np.random.uniform(0, 1, n_components)[:, np.newaxis]
        # np.array of size (n_components, 1)

        # Initialized with identity.
        self._sigma = np.stack([np.eye(n_dims) for _ in range(n_components)])
        # np.array of size (n_components, n_dims, n_dims)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        
        print("initialization with data...")
        initial_mean = x[np.random.choice(x.shape[0], self._n_components, replace=False), :]
        self._mu = initial_mean
        '''
        print("initialization with kmeans...")
        kmeans = KMeans(n_clusters=self._n_components, random_state=0).fit(x)
        self._mu = kmeans.cluster_centers_
        '''
        for _ in range(self._max_iter):
            # print(_," step mu:",self._mu)
            # print(" pi:", self._pi, " sigma:", self._sigma)
            
            z_ik = self._e_step(x)  # E step
            print("done E step")
            self._m_step(x, z_ik)   # M step
            print("done M step")


    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = self.get_posterior(x)

        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        N = x.shape[0]
       # print(z_ik.shape, x.shape, (N * self._pi).shape)
        self._pi = np.sum(z_ik, axis=0) / N # sum N points for each component 
        self._mu = np.dot(z_ik.T, x) / (N * self._pi)[:, None] # (n_component, n_dim) / (n_component, 1)
        
        self._sigma = np.zeros((self._n_components, self._n_dims, self._n_dims))
        for j in range(self._n_components):
            for i in range(N):
                x_mu = (x[i] - self._mu[j])[:, np.newaxis]
                #if i % 20 == 0:
                    #print(np.dot(x_mu.T, x_mu).shape)
                self._sigma[j] += z_ik[i, j] * np.dot(x_mu, x_mu.T)
            self._sigma[j] /= (N * self._pi[j])
            self._sigma[j] += np.eye(len(self._sigma[j]))*self._reg_covar
            #print(self._sigma[j])

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).
        """
        N = x.shape[0]
        ret = np.zeros((N, self._n_components))
        for j in range(self._n_components):
            ret[:, j] = self._multivariate_gaussian(x, self._mu[j], self._sigma[j])
        return ret

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        N = x.shape[0]
        marginal = np.zeros(N)
        for j in range(self._n_components):
            marginal += self._pi[j] * self._multivariate_gaussian(x, self._mu[j], self._sigma[j])
        return marginal

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = np.zeros((x.shape[0], self._n_components)) # Aij in lecture
        nomi = self.get_conditional(x)
        domi = self.get_marginals(x)

        for j in range(self._n_components):
            result = (self._pi[j] * nomi[:, j]) / domi
            z_ik[:, j] = result  # put result back into single column
        
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, 1)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.fit(x) # first perform unsupervised clustering
        cluster = np.argmax(self.get_posterior(x), axis=1) # get the MAP of data
        global label_dict
        label_dict = dict(zip(set(y), [i for i in range(len(set(y)))]))
        
        count = np.zeros((self._n_components ,len(set(y)))) 
        print(count.shape)

        for i in range(len(x)):
            count[cluster[i], int(label_dict[y[i]])] += 1

        self.cluster_label_map = np.argmax(count, axis=1)

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        global label_dict

        p = dict(zip(label_dict.values(),label_dict.keys()))
        z_ik = self.get_posterior(x)
        cluster = np.argmax(z_ik, axis=1)
        y_hat = list()
        for element in cluster:
            y_hat.append(p[self.cluster_label_map[element]])


        return np.array(y_hat)
