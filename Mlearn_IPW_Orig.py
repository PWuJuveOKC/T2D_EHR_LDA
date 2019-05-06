import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import pairwise_distances
from skll.metrics import kappa



class MatchOLearn:

  def __init__(self, kernel='rbf', C=1.0, gamma=1.0, propensity=0.5, n_jobs=1, metric='l2'):
    self.kernel = kernel
    self.C = C
    self.gamma = gamma
    self.propensity = propensity
    self.n_jobs = n_jobs
    self.metric = metric


  def fit(self, X, Q, A, match, learn, ipw, K=1, g=1, type='continuous'):

    X1 = X[np.where(A == -1)]
    X2 = X[np.where(A == 1)]

    A1 = A[A == -1]
    A2 = A[A == 1]

    Q1 = Q[np.where(A == -1)]
    Q2 = Q[np.where(A == 1)]

    ipw1 = ipw[np.where(A == -1)]
    ipw2 = ipw[np.where(A == 1)]

    Q1_K = np.tile(Q1, K)
    Q2_K = np.tile(Q2, K)

    A1_K = np.tile(A1, K)
    A2_K = np.tile(A2, K)

    ipw1_K = np.tile(ipw1, K)
    ipw2_K = np.tile(ipw2, K)

    X1_K = np.tile(X1, (K, 1))
    X2_K = np.tile(X2, (K, 1))

    if type == 'continuous':
        # Match for X1
        X2_pair_value = []
        for i in range(len(X1)):
            sim = pairwise_distances(X1[i, match].reshape(1, -1), X2[:, match], metric=self.metric)
            ind = sim[0].argsort()[:K]
            X2_pair_value.append(Q2[ind])
            Q1_paired_array = np.asarray(np.transpose(X2_pair_value)).reshape(-1)

        # Match for X2
        X1_pair_value = []
        for i in range(len(X2)):
            sim = pairwise_distances(X2[i, match].reshape(1, -1), X1[:, match], metric=self.metric)
            ind = sim[0].argsort()[:K]
            X1_pair_value.append(Q1[ind])
            Q2_paired_array = np.asarray(np.transpose(X1_pair_value)).reshape(-1)

    elif type == 'ordinal':
        X2_pair_value = []
        for i in range(len(X1)):
            sim_kappa_X1 = [kappa(X1[i], X2[j], weights='quadratic') for j in range(len(X2))]
            ind = (-np.array(sim_kappa_X1)).argsort()[:K]
            X2_pair_value.append(Q2[ind])
        Q1_paired_array = np.asarray(np.transpose(X2_pair_value)).reshape(-1)

        # Match for X2
        X1_pair_value = []
        for i in range(len(X2)):
            sim_kappa_X2 = [kappa(X2[i], X1[j], weights='quadratic') for j in range(len(X1))]
            ind = (-np.array(sim_kappa_X2)).argsort()[:K]
            X1_pair_value.append(Q1[ind])
        Q2_paired_array = np.asarray(np.transpose(X1_pair_value)).reshape(-1)


    X_MatchO = np.concatenate((X1_K, X2_K), axis=0)

    DIFF = np.append((Q1_K - Q1_paired_array), (Q2_K - Q2_paired_array))
    if g > 0:
        self.Weights = abs(DIFF) ** g
    elif g == 0:
        self.Weights = np.sign(DIFF)

    self.new_label = np.append(A1_K, A2_K) * np.sign(DIFF + np.random.uniform(-1e-5, 1e-5))

    self.Weights = self.Weights * np.append(1/ipw1_K, 1/ipw2_K)

    self.clf = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel)
    self.clf.fit(X_MatchO[:, learn], self.new_label, sample_weight=self.Weights)
    if self.kernel == 'linear':
        self.coef_ = self.clf.coef_

    return self

  def predict(self, X):

    classification = self.clf.predict(X)

    return classification

  def estimate(self, X, Q, A, learn, ipw, normalize=True):

      classification = self.clf.predict(X[:,learn])
      if self.propensity == 'obs':
          logist = linear_model.LogisticRegression()
          logist.fit(X, A)
          prob = logist.predict_proba(X)[:, 1]
      else:
          prob = self.propensity
      PS = (prob * A + (1 - A) / 2) * ipw

      Q0 = Q[np.where(A == classification)]
      PS0 = PS[np.where(A == classification)]

      if not normalize:
        est_Q = np.sum(Q0 / PS0) / len(Q)
      elif normalize:
          est_Q = np.sum(Q0 / PS0) / np.sum(1 / PS0)

      return est_Q
