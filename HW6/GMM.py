from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from six.moves import range
from scipy import stats
from matplotlib import pyplot as plt

X = np.loadtxt('iris.data', dtype='object', delimiter=',')
Y = X[:,-1]
X = X[:, :-1].astype('f')
#X.shape, Y.shape, Y.dtype #((150, 4), (150,), dtype('O'))

def gmm(X, n_classes, n_iter):
    # TODO fill in your code here
    
    #Initialize covariance matrix
    Pi = np.array([1./ n_classes] *  n_classes)
    mean = np.random.rand(n_classes, X.shape[1])
    
    cov = np.array([np.eye(X.shape[1])] * n_classes)


    for k in range(n_iter):
        print('K=', k)
        # E-step
        density = np.empty((X.shape[0], n_classes))
        for i in range(n_classes):
            density[:,i] = Pi[i] * stats.multivariate_normal.pdf(X, mean[i], cov[i], allow_singular=True)
        posterior = density / density.sum(axis=1).reshape(-1, 1)

        # M-step
        Pi_hat = posterior.sum(axis=0) / posterior.sum()
        print(Pi_hat)
        mean_hat = np.zeros((n_classes, X.shape[1]))
        cov_hat = []
        for i in range(n_classes):
            mean_hat[i] = np.average(X, axis=0, weights=posterior[:, i])
            cov_hat.append(np.cov(X - mean_hat[i], rowvar=0, aweights=posterior[:, i]))

        # Update
        cov = cov_hat
        mean = mean_hat
        Pi = Pi_hat

    class_assignments = np.argmax(posterior, axis=1)

    return class_assignments, mean, cov

class_assignments, mean, cov = gmm(X, 3, 100)
print(class_assignments)