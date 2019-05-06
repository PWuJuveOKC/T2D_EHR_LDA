import numpy as np
from sklearn import linear_model
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")



class QLearn:

  def __init__(self, alpha, propensity=0.5):
    self.alpha = alpha
    self.propensity = propensity


  def fit(self, X, Q, A, ipw):
    Qreg = linear_model.Ridge(alpha=self.alpha)

    self.p = X.shape[1]
    QL_Int = X * np.transpose(np.tile(A, (self.p, 1)))
    Qreg.fit(np.concatenate((X, QL_Int, A.reshape(-1, 1)), axis=1), Q, sample_weight=1/ipw)
    self.coeff = Qreg.coef_

    return self

  def predict(self,X):
    classification = np.sign(sum(np.transpose(self.coeff[self.p:(2 * self.p)] * np.array(X)) +
                                 np.ones(len(X))) * self.coeff[2 * self.p] + np.random.uniform(-1e-5,1e-5))
    return classification

  def estimate(self, X, Q, A, learn, ipw, normalize=True):
      classification = np.sign(sum(np.transpose(self.coeff[self.p:(2 * self.p)] * np.array(X[:,learn]))) +
                               np.ones(len(X[:,learn])) * self.coeff[2 * self.p] + np.random.uniform(-1e-5,1e-5))

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






