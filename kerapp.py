# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:28:48 2017
Compare different types of kernel approximations
@author: afsar
"""

import matplotlib.pyplot as plt
import itertools
from sklearn import svm
import numpy as np
from sklearn.kernel_approximation import RBFSampler,Nystroem
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures

class RBFSampler2(RBFSampler):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.

    It implements a variant of Random Kitchen Sinks.[1] 
    It differs from RBFSampler in that it uses both sin and cos and does not use
    an offset. As a consequence, it will generate 2*n_components features.

    Read more in the :ref:`User Guide <rbf_kernel_approx>`.

    Parameters
    ----------
    gamma : float
        Parameter of RBF kernel: exp(-gamma * x^2)

    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : {int, RandomState}, optional
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator.

    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.

    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (http://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    """    
    def transform(self, X, y=None):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'random_weights_')
        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        out = np.hstack((np.cos(projection),np.sin(projection)))
        out *= np.sqrt(2.) / np.sqrt(2*self.n_components)
        return out
        
from scipy.linalg import norm, qr
class ORFSampler(RBFSampler):
    """Approximates feature map of an RBF kernel by Orthogonal random features


    Parameters
    ----------
    gamma : float
        Parameter of RBF kernel: exp(-gamma * x^2)

    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : {int, RandomState}, optional
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator.

    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    adapted from: https://github.com/NICTA/revrand/blob/master/revrand/basis_functions.py
    [1] Orthogonal Random Features Felix Xinnan Yu*, ; Ananda Theertha Suresh, ; Krzysztof Choromanski, ; Dan Holtmann-Rice, ; Sanjiv Kumar, Google
    """

    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """

        X = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        self.random_weights_ =  _weightsamples(self.n_components,n_features,random_state)
        return self

    def transform(self, X, y=None):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        check_is_fitted(self, 'random_weights_')

        X = check_array(X, accept_sparse='csr')
        
        sigma = 1./np.sqrt(2*self.gamma)        
        #WX = np.dot(X, self.W / sigma)
        WX = safe_sparse_dot(X, self.random_weights_/sigma)
        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(self.n_components)

        
def _weightsamples(n,d,random_state):
        reps = int(np.ceil(n / d))
        Q = np.empty((d, d*reps))

        for r in range(reps):
            W = random_state.normal(size=(d, d))
            Q[:, (r * d):((r + 1) * d)] = qr(W)[0]

        S = np.sqrt(random_state.chisquare(df=d, size=d))
        weights = np.diag(S).dot(Q[:, :n])
        return weights

n = 1000
Xtr = np.vstack((np.random.randn(n,2)+1,np.random.randn(n,2)-1))
Ytr = np.ones(2*n); Ytr[n:]=-1;

gamma = 0.5
C = 100
nc = 100

rbf = RBFSampler(gamma = gamma,n_components = nc) 
rbf.fit(Xtr)
orf = ORFSampler(gamma = gamma,n_components = nc) 
orf.fit(Xtr)
# the random state needs to be specified so that the test and train features are the same
clfa = svm.LinearSVC( C = C, loss='hinge')
clfa.fit(rbf.transform(Xtr), Ytr)

clfo = svm.LinearSVC( C = C, loss='hinge')
clfo.fit(orf.transform(Xtr), Ytr)

clfa_sgd = SGDClassifier(alpha = 1./(C*len(Ytr)), loss='hinge',n_iter=10,learning_rate ='optimal')
clfa_sgd.fit(rbf.transform(Xtr), Ytr)

clfp_sgd = SGDClassifier(alpha = 0.1, loss='hinge',n_iter=40,learning_rate ='optimal')
poly = PolynomialFeatures(degree=3)
poly.fit_transform(Xtr)
clfp_sgd.fit(poly.transform(Xtr), Ytr)

rbf2 = RBFSampler2(gamma = gamma,n_components = nc/2) 
rbf2.fit(Xtr)

# the random state needs to be specified so that the test and train features are the same
clfa2 = svm.LinearSVC(C = C, loss='hinge')
clfa2.fit(rbf2.transform(Xtr), Ytr)


nys = Nystroem(kernel='rbf', gamma=gamma, coef0=1, degree=3, kernel_params=None, n_components=nc, random_state=None)
nys.fit(Xtr)
clf3 = svm.LinearSVC( C = C, loss='hinge')
clf3.fit(nys.transform(Xtr), Ytr)

clf = svm.SVC(kernel='rbf', C = C, gamma = gamma)
clf.fit(Xtr, Ytr)

plt.close('all')
npts = 50
xm,xM = np.min(Xtr),np.max(Xtr)
x = np.linspace(xm,xM,npts)
y = np.linspace(xm,xM,npts)
t = np.array(list(itertools.product(x,y)))

def plotit(z,title):
    plt.figure()    
    z = np.reshape(z,(npts,npts))
    plt.imshow(z)    
    plt.contour(z,[-1,0,+1],linewidths = [1,2,1],colors=('k'),extent=[xm,xM,xm,xM], label='f(x)=0')
    plt.imshow(np.flipud(z), extent = [xm,xM,xm,xM], cmap=plt.cm.Purples); plt.colorbar()
    plt.scatter(Xtr[Ytr==1,0],Xtr[Ytr==1,1],marker = 's', c = 'r', s = 10)
    plt.scatter(Xtr[Ytr==-1,0],Xtr[Ytr==-1,1],marker = 'o',c = 'g', s = 10)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis([xm,xM,xm,xM])    
    plt.grid()
    plt.title(title)
    plt.show()
    
plotit(clfa.decision_function(rbf.transform(t)),'RBF Appox')
plotit(clfo.decision_function(orf.transform(t)),'ORF Appox')
plotit(clfa_sgd.decision_function(rbf.transform(t)),'RBF Appox SGD')
plotit(clfa2.decision_function(rbf2.transform(t)),'RBF-2 Appox')
plotit(clf3.decision_function(nys.transform(t)),'Nys Appox')
plotit(clfp_sgd.decision_function(poly.transform(t)),'Poly SGD Appox')
plotit(clf.decision_function(t),'True SVM')

