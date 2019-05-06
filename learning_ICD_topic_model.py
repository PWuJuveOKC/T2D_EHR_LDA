import pandas as pd
from Mlearn_IPW_Orig import MatchOLearn
from Olearn_IPW_Orig import OLearn
from Qlearn_IPW import QLearn
import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
from sklearn.utils import shuffle

num_cores = multiprocessing.cpu_count()
normal = True
met = 'mahalanobis'
ps = 'obs'
Adjust = True
baseline = 'high'
dat = pd.read_csv('Datasets/dat_ICD_{}.csv'.format(baseline))

def Learning(t):

    np.random.seed(t)
    dat_new = shuffle(dat)

    Q = - np.array(dat_new['ICD_post_outcome'] )
    A = np.array(dat_new['TRT'])

    ipw = np.array(dat_new['IPW'])
    del (dat_new['patient_id'])
    del dat_new['TRT']
    del dat_new['IPW']
    del dat_new['ICD_post_outcome']
    X = np.array(dat_new.iloc[:, :])

    test_size = 0.5

    Q_train = Q[:-int(test_size * len(Q))]
    Q_test = Q[-int(test_size * len(Q)):]

    A_train = A[:-int(test_size * len(Q))]
    A_test = A[-int(test_size * len(Q)):]

    X_train = X[:-int(test_size * len(Q))]
    X_test = X[-int(test_size * len(Q)):]

    ipw_train = ipw[:-int(test_size * len(Q))]
    ipw_test = ipw[-int(test_size * len(Q)):]

    Cs = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    gammas = [0.001, 0.01, 0.1, 1, 2, 5, 10]
    kernels = ['linear', 'rbf']
    alphas = [0.01, 0.1, 1, 10, 100]

    psmatch_ind = list(range(X.shape[1] - 3))
    psmatch_ind.remove(22)
    psmatch_ind.remove(23)

    learn_ind = list(range(31))
    learn_ind.remove(22)
    learn_ind.remove(23)

    match_ind = [2+10, 3+10, 4+10, 5+10, 6+10, 7+10, 8+10, 9+10, 10+10, 11+10, 12+10,
                 X.shape[1]-3,X.shape[1]-2, X.shape[1]-1]

    all_OL = []
    for kernel in kernels:
        for C in Cs:
            for gamma in gammas:
                model_OL1 = OLearn(C=C, gamma=gamma, kernel=kernel, propensity=ps)
                model_OL1.fit(X_train, Q_train, A_train, psmatch=np.array(range(X.shape[1]-3)),
                              learn= np.array(learn_ind), ipw=ipw_train)
                est_OL1 = -1 * model_OL1.estimate(X_test[:, psmatch_ind], Q_test, A_test,
                                                  learn= np.array(learn_ind),normalize=normal, ipw=ipw_test)

                model_OL2 = OLearn(C=C, gamma=gamma, kernel=kernel, propensity=ps)
                model_OL2.fit(X_test, Q_test, A_test, psmatch=np.array(psmatch_ind),
                              learn= np.array(learn_ind), ipw=ipw_test)
                est_OL2 = -1 * model_OL2.estimate(X_train[:, psmatch_ind], Q_train, A_train,
                                                  learn= np.array(learn_ind),normalize=normal, ipw=ipw_train)
                all_OL.append(np.mean([est_OL1, est_OL2]))

    all_QL = []
    for alpha in alphas:
        model_QL1 = QLearn(alpha=alpha,propensity=ps)
        model_QL1.fit(X_train[:,np.array(learn_ind)], Q_train, A_train,ipw=ipw_train)
        est_QL1 = -1 * model_QL1.estimate(X_test[:, psmatch_ind],Q_test, A_test,
                                          learn= np.array(learn_ind),ipw=ipw_test, normalize=normal)

        model_QL2 = QLearn(alpha=alpha,propensity=ps)
        model_QL2.fit(X_test[:,np.array(learn_ind)], Q_test, A_test, ipw=ipw_test)
        est_QL2 = -1 * model_QL2.estimate(X_train[:, psmatch_ind], Q_train, A_train,
                                          learn= np.array(learn_ind),ipw=ipw_train,normalize=normal)
        all_QL.append(np.mean([est_QL1, est_QL2]))

    all_MatchO = []
    for kernel in kernels:
        for C in Cs:
            for gamma in gammas:
                model_MO1 = MatchOLearn(C=C, gamma=gamma, kernel=kernel, propensity=ps, metric=met)
                model_MO1.fit(X_train, Q_train, A_train, ipw = ipw_train, match=np.array(match_ind),
                              learn=np.array(learn_ind), g=1, K=1)
                est_MatchO1 = -1 * model_MO1.estimate(X_test[:,psmatch_ind], Q_test, A_test, ipw = ipw_test,
                                                      learn= np.array(learn_ind),normalize=normal)

                model_MO2 = MatchOLearn(C=C, gamma=gamma, kernel=kernel, propensity=ps,metric=met)
                model_MO2.fit(X_test, Q_test, A_test,ipw =ipw_test, match=np.array(match_ind),
                              learn=np.array(learn_ind), g=1, K=1)
                est_MatchO2 = -1 * model_MO2.estimate(X_train[:,psmatch_ind], Q_train, A_train, ipw =ipw_train,
                                                      learn= np.array(learn_ind),normalize=normal)
                all_MatchO.append(np.mean([est_MatchO1, est_MatchO2]))

    return [np.min(all_QL), np.min(all_OL), np.min(all_MatchO)]


results = {}
iters = 100
start = time.time()
results = Parallel(n_jobs= (num_cores-1))(delayed(Learning)(t) for t in range(iters))
cost = time.time() - start
print("Time consumed: ", cost)

print('Mean: ', np.mean(np.array(results)[:, 0]), np.mean(np.array(results)[:, 1]), np.mean(np.array(results)[:, 2])),
print('Median: ',np.median(np.array(results)[:, 0]), np.median(np.array(results)[:, 1]), np.median(np.array(results)[:, 2])),
print('Std: ', np.std(np.array(results)[:, 0]), np.std(np.array(results)[:, 1]), np.std(np.array(results)[:, 2]))

print("Q: 25%, 50%, 75%", np.percentile(np.array(results)[:,0], (25, 50, 75)))
print("O: 25%, 50%, 75%", np.percentile(np.array(results)[:,1], (25, 50, 75)))
print("M: 25%, 50%, 75%", np.percentile(np.array(results)[:,2], (25, 50, 75)))
