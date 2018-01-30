from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import argparse
import sys
import os
import csv

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.sum(np.diag(C))/np.sum(C)

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C) / np.sum(C, axis=1)
def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C) / np.sum(C, axis=0)



def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    feats = np.load(filename)
    X_train, X_test, y_train, y_test = train_test_split(feats['arr_0'][:, 0:173], feats['arr_0'][:, 173], test_size=0.2,
                                                        random_state=10)
    #TODO: Check NAN feature mostly related to norm
    X_train = np.nan_to_num(X_train)
    X_test  = np.nan_to_num(X_test)

    #SVM with linear kernel
    print("Using svm(linear classiifier)")
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    print("using svm(linear) predict")
    sl_y_prediction = svm_linear.predict(X_test)
    sl_cm = confusion_matrix(y_test,sl_y_prediction)
    sl_acc = accuracy(sl_cm)
    sl_re = recall(sl_cm)
    sl_pre = precision(sl_cm)
    print("acc:{},\nre:{},\npre:{}".format(sl_acc,sl_re,sl_pre))

    #SVM with rbf
    print("Using svm(rbf classiifier)")
    svm_rbf = SVC(kernel='rbf', gamma=2)
    svm_rbf.fit(X_train, y_train)
    print("using svm(rbf) predict")
    sr_y_prediction = svm_rbf.predict(X_test)
    sr_cm = confusion_matrix(y_test, sr_y_prediction)
    sr_acc = accuracy(sr_cm)
    sr_re = recall(sr_cm)
    sr_pre = precision(sr_cm)
    print("acc:{},\nre:{},\npre:{}".format(sr_acc, sr_re, sr_pre))

    #Random forest
    print("Using forest classifier")
    rfc = RandomForestClassifier(n_estimators=10,max_depth=5)
    rfc.fit(X_train, y_train)
    print("using rfc predict")
    rfc_y_prediction = rfc.predict(X_test)
    rfc_cm = confusion_matrix(y_test, rfc_y_prediction)
    rfc_acc = accuracy(rfc_cm)
    rfc_re = recall(rfc_cm)
    rfc_pre = precision(rfc_cm)
    print("acc:{},\nre:{},\npre:{}".format(rfc_acc, rfc_re, rfc_pre))

    #MLP
    print("Using MLP classifier")
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(X_train, y_train)
    print("using mlp predict")
    mlp_y_prediction = mlp.predict(X_test)
    mlp_cm = confusion_matrix(y_test, mlp_y_prediction)
    mlp_acc = accuracy(mlp_cm)
    mlp_re = recall(mlp_cm)
    mlp_pre = precision(mlp_cm)
    print("acc:{},\nre:{},\npre:{}".format(mlp_acc, mlp_re, mlp_pre))

    #AdaBoost
    print("Using AdaBoost classifier")
    adbc = AdaBoostClassifier()
    adbc.fit(X_train, y_train)
    print("using adbc predict")
    adbc_y_prediction = adbc.predict(X_test)
    adbc_cm = confusion_matrix(y_test, adbc_y_prediction)
    adbc_acc = accuracy(adbc_cm)
    adbc_re = recall(adbc_cm)
    adbc_pre = precision(adbc_cm)
    print("acc:{},\nre:{},\npre:{}".format(adbc_acc, adbc_re, adbc_pre))

    iBest = np.array([sl_acc, sr_acc, rfc_acc, mlp_acc, adbc_acc]).argmax() + 1
    #Write to csv
    with open('a1_3.1.csv','w', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',')
        file_writer.writerow([1,sl_acc]+sl_re.tolist()+sl_pre.tolist()
                             + sl_cm[0].tolist() +sl_cm[1].tolist()
                             + sl_cm[2].tolist() +sl_cm[3].tolist())
        file_writer.writerow([2, sr_acc] + sr_re.tolist() + sr_pre.tolist()
                             + sr_cm[0].tolist() + sr_cm[1].tolist()
                             + sr_cm[2].tolist() + sr_cm[3].tolist())
        file_writer.writerow([3, rfc_acc] + rfc_re.tolist() + rfc_pre.tolist()
                             + rfc_cm[0].tolist() + rfc_cm[1].tolist()
                             + rfc_cm[2].tolist() + rfc_cm[3].tolist())
        file_writer.writerow([4, mlp_acc] + mlp_re.tolist() + mlp_pre.tolist()
                             + mlp_cm[0].tolist() + mlp_cm[1].tolist()
                             + mlp_cm[2].tolist() + mlp_cm[3].tolist())
        file_writer.writerow([5, adbc_acc] + adbc_re.tolist() + adbc_pre.tolist()
                             + adbc_cm[0].tolist() + adbc_cm[1].tolist()
                             + adbc_cm[2].tolist() + adbc_cm[3].tolist())

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''




    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
