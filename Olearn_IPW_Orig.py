import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model



class OLearn:

  def __init__(self, kernel='rbf', C=1.0, gamma=1.0, propensity=0.5, n_jobs=1):
    self.kernel = kernel
    self.C = C
    self.gamma = gamma
    self.propensity = propensity
    self.n_jobs = n_jobs


  def fit(self, X, Q, A, psmatch, learn, ipw):
    Q = Q - np.min(Q)
    logist = linear_model.LogisticRegression(n_jobs =  self.n_jobs)

    if self.propensity == 'obs':
        logist.fit(X[:,psmatch], A)
        prob = logist.predict_proba(X[:,psmatch])[:, 1]
    else:
        prob = self.propensity
    PS = prob * A + (1 - A) / 2
    self.new_label = A
    self.Weights = np.array(Q / PS) / ipw
    self.clf = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel)
    self.clf.fit(X[:,learn],self.new_label,sample_weight=self.Weights)

    return self

  def predict(self,X):

    classification = self.clf.predict(X)
    return classification

  def estimate(self,X,Q,A,learn,ipw,normalize=True):
      classification =  self.clf.predict(X[:,learn])
      if self.propensity == 'obs':
          logist = linear_model.LogisticRegression()
          logist.fit(X, A)
          prob = logist.predict_proba(X)[:, 1]
      else:
          prob = self.propensity
      PS = (prob * A + (1 - A) / 2 )* ipw

      Q0 = Q[np.where(A == classification)]
      PS0 = PS[np.where(A == classification)]

      if not normalize:
        est_Q = np.sum(Q0 / PS0) / len(Q)
      elif normalize:
          est_Q = np.sum(Q0 / PS0) / np.sum(1 / PS0)

      return est_Q

