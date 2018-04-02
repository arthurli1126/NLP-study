from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.misc import logsumexp

#dataDir = '/u/cs401/A3/data/'
dataDir = '../data/'
class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    prec = preComputedForM[m]
    sigma = np.diag(myTheta.Sigma[:,:,m])
    first_total = 0
    first_part= np.sum(0.5*(x**2)*(sigma**(-1)))
    sec_part = np.sum(myTheta.mu[m]*x*(sigma**(-1)))
    first_total += sec_part - first_part
    return first_total-prec
    
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


def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    T= log_Bs.shape[1]
    part1 = np.log(np.tile(myTheta.omega.reshape(-1,1),(1,T)))
    part2 =log_Bs
    return logsumexp(part1 + part2 )



    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    T = X.shape[0]
    D = X.shape[1]
    myTheta = theta( speaker, M, X.shape[1] )
    mat_1 = np.zeros([M,T])
    mat_2 = np.zeros([M,T])

    #initialization
    myTheta.omega = np.zeros([1,M]) + 1/M
    myTheta.mu =np.zeros([M,D])
    myTheta.Sigma = np.zeros([D,D,M])

    for i in range(M):
        myTheta.Sigma[:,:,i] = np.identity(D)
        myTheta.mu[i] = X[random.randint(1, T)]

    #train
    prev_ll = float("-inf")
    imp = epsilon

    curr_i = 0
    while curr_i<maxIter and imp>= epsilon:
        pre_c = pre_com(myTheta)
        for m in range(M):
            for t in range(T):
                #print("m: {}, t: {}".format(m,t))
                mat_1[m][t] = log_b_m_x(m,X[t],myTheta,pre_c)
                mat_2[m][t] = log_p_m_x(m,X[t],myTheta)
        print("testing")
        ll = logLik(mat_1,myTheta)
        print("ll:{}".format(ll))
        myTheta = update_param(myTheta,X,mat_2,M)
        imp = ll-prev_ll
        print(imp)
        prev_ll = ll
        curr_i+=1

    return myTheta


def update_param(theta, X, p_m_x, M):
    T = X.shape[0]
    D = X.shape[1]

    print(p_m_x)
    #omega
    p_sum = np.sum(p_m_x, axis=1)
    theta.omega = (p_sum /T).reshape(1,M)
    #mu
    p_sum_rep = np.tile(p_sum,(D,1))
    X_com_conj_trans = X.conj().transpose()
    theta.mu = ((X_com_conj_trans.dot(p_m_x.conj().transpose())) / p_sum_rep)
    #variance
    mu_sq = theta.mu*theta.mu
    X_sq_con_trans = (X*X).conj().transpose()
    sum_p_x_sq = X_sq_con_trans.dot(p_m_x.conj().transpose())
    var = (sum_p_x_sq / p_sum_rep) - mu_sq

    theta.mu = theta.mu.conj().transpose()

    for m in range(M):
        theta.Sigma[:,:,m] = np.diag(var[:,m])

    return theta

def test( mfcc, correctID, models, k=5 ):
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
    print ('TODO')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    #todo: change paramter for test
    d = 13
    k = 4  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)
            #todo needed for test
            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

