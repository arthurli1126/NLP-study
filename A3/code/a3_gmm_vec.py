from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.misc import logsumexp
import argparse


#dataDir = '/u/cs401/A3/data/'
dataDir = '../data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def pre_com(myTheta):

    d = myTheta.Sigma.shape[0]
    M = myTheta.omega.shape[1]
    sec = (d / 2) * np.log(2 * np.pi)
    first = []
    for m in range(M):
        cov = np.diag(myTheta.Sigma[:,:,m])
        u_over_sig = np.sum((myTheta.mu[m]*myTheta.mu[m])/(2 * cov))
        third = 0.5 * np.log(np.prod(cov))
        first.append(u_over_sig+sec+third)
    return first

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    omega_m_part = myTheta.omega[:,m]
    # sigma_m = myTheta.Sigma[m]
    m_size = len(myTheta.mu)

    #todo use prec function
    pre_c_m = pre_com(myTheta)

    log_omega_m = np.log(omega_m_part)
    log_bm = log_b_m_x(m,x,myTheta,pre_c_m)
    #sum_log_mu = np.sum(np.log(myTheta.mu))

    b = np.array([ log_b_m_x(i,x,myTheta, pre_c_m) for i in range(m_size)])
    #to use logsumexp
    log_sum_b_omega = logsumexp(np.log(myTheta.omega) + b)

    return log_omega_m + log_bm - log_sum_b_omega


def log_b_m_x(M, X, myTheta):
    T = X.shape[0]
    D = X.shape[1]
    log_b = np.zeros((T,M))

    for m in range(M):
        mu_m = myTheta.mu[:,m]
        #t*D
        mu_m_rep = np.tile(mu_m.transpose(),(T,1))
        cov_m = np.diag(myTheta.Sigma[:,:,m])
        cov_m_rep = np.tile(cov_m.transpose(), (T,1))
        log_b[:,m] = logPdf(X, mu_m_rep, cov_m_rep)
    return log_b


def logPdf(X, mu, cov):

    D = X.shape[1]
    x_m_mu = X - mu
    part2 = (x_m_mu * x_m_mu) / cov
    log_b_t = -0.5 * (part2 + np.log(2*np.pi*cov))
    log_b = np.sum(log_b_t, axis=1)

    return log_b



def logLik(X, myTheta, M):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    T = X.shape[0]
    D = X.shape[1]

    log_bs = log_b_m_x(M,X,myTheta) #TxM

    log_omega_rep = np.tile(np.log(myTheta.omega), (T,1))
    L = log_bs+log_omega_rep
    last = logsumexp(log_bs + log_omega_rep, axis=1)
    #test = logsumexp(log_bs + log_omega_rep, axis=0)
    L = np.sum(last)
    last = last.reshape(-1,1)
    #test = test.reshape(1, -1)
    log_p_m_x = log_omega_rep + log_bs - np.tile(last, (1,M))
    return L, log_p_m_x


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    T = X.shape[0]
    D = X.shape[1]
    myTheta = theta(speaker, M, X.shape[1])

    # initialization
    myTheta.omega = np.zeros([1, M]) + 1 / M
    myTheta.mu = np.zeros([D, M])
    myTheta.Sigma = np.zeros([D, D, M])

    for i in range(M):
        myTheta.Sigma[:, :, i] = np.identity(D)
        myTheta.mu[:,i] = X[random.randint(0, T-1)]

    # train
    prev_ll = float("-inf")
    imp = float("inf")

    epsilon = epsilon
    curr_i = 0
    while curr_i < maxIter and imp > epsilon:
        L,pmx = logLik(X,myTheta,M)
        print("ll:{}".format(L))

        myTheta = update_param(myTheta, X, pmx, M)
        imp = L - prev_ll
        print("iter {}，imp : {}".format(curr_i,imp))
        prev_ll = L
        curr_i += 1

    return myTheta


def update_param(theta, X, p_m_x, M):
    T = X.shape[0]
    D = X.shape[1]
    real_pmx = np.exp(p_m_x)

    # omega
    p_sum = np.sum(real_pmx, axis=0)
    theta.omega = (p_sum / T)
    #print(theta.omega)
    # mu
    p_sum_rep = np.tile(p_sum, (D, 1))
    Xtrans = X.transpose()
    theta.mu = (Xtrans.dot(real_pmx)) / p_sum_rep
    # variance
    mu_sq = theta.mu * theta.mu
    X_sq_trans = (X * X).transpose()
    sum_p_x_sq = X_sq_trans.dot(real_pmx)

    var = (sum_p_x_sq / p_sum_rep) - mu_sq

    for m in range(M):
        theta.Sigma[:, :, m] = np.diag(var[:, m])
    return theta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    M = models[0].mu.shape[1]
    lls = []
    #print(M)
    for i in range(len(models)):
        L, p_m_x = logLik(mfcc,models[i],M)
        lls.append(L)
    #print(lls)
    lls = np.array(lls)

    top_k_index = np.argsort(lls)[-k:]
    print(models[correctID].name)
    for i in top_k_index:
        print("{} {}".format(models[i].name,lls[i]))
    bestModel = top_k_index[-1]
    print("\n")
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='paramters .')

    parser.add_argument("-m", "--M", help="size of M", required=True)
    parser.add_argument("-i", "--i", help="size of iter", required=True)
    args = parser.parse_args()



    trainThetas = []
    testMFCCs = []
    # todo: change paramter for test
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    #M = 8
    M = int(args.M)
    maxIter = int(args.i)
    epsilon = 10000

    #maxIter = 8
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print(accuracy)

