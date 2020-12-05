import pandas as pd
import matplotlib as plt
import numpy as np
from collections import OrderedDict
import seaborn as sns
from termcolor import colored

from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn

NOTION_ELECTRODES = ["CP6","F6","C4","CP4","CP3","F5","C3","CP5"]
MUSE_ELECTRODES = ['TP9', 'AF7', 'AF8', 'TP10']
MUSE_FREQUENCY = 256
NOTION_FREQUENCY = 250

from tools import *

ALPHA_MARKERS = ["eyes-open", "eyes-closed"]
SMR_MARKERS = ["rest","visualisation","movement"]
N170_MARKERS = ['house','face']
P300_MARKERS = ['red', 'blue' ,'press']

def N170_test(session_data):
    markers = N170_MARKERS
    epochs = get_session_erp_epochs(session_data, markers)
    conditions = OrderedDict()
    for i in range(len(markers)):
        conditions[markers[i]] = [i+1]
   
    clfs = OrderedDict()
    
    clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
    clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
    clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(), LogisticRegression())
    clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(estimator='oas'), MDM())
    clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression())
    clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())
    methods_list = ['Vect + LR','Vect + RegLDA','ERPCov + TS','ERPCov + MDM','XdawnCov + TS','XdawnCov + MDM']
    # format data
    epochs.pick_types(eeg=True)
    X = epochs.get_data() * 1e6
    times = epochs.times
    y = epochs.events[:, -1]

    # define cross validation 
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25, 
                                random_state=42)

    # run cross validation for each pipeline
    auc = []
    methods = []
    print('Calcul in progress...')
    for m in clfs:
        try:

            res = cross_val_score(clfs[m], X, y==2, scoring='roc_auc', 
                                  cv=cv, n_jobs=-1)
            auc.extend(res)
            methods.extend([m]*len(res))
        except Exception:
            print("exception")
        
    ## Plot Decoding Results

    results = pd.DataFrame(data=auc, columns=['AUC'])
    results['Method'] = methods
    n_row,n_column = results.shape
    auc_means = []
    for method in methods_list:
        auc = []
        for i in range(n_row):
            if results.loc[i,'Method']== method:
                auc.append(results.loc[i,'AUC'])
        auc_means.append(np.mean(auc))
    counter = 0
    for i in range(len(methods_list)):
        color = 'green' if auc_means[i]>=0.7 else 'red'
        counter = counter +1 if auc_means[i]>=0.7 else counter
        
    return counter > 0, counter
    
    
def P300_test(session_id):
    markers = P300_MARKERS
    epochs = get_session_erp_epochs(session_id, markers)
    conditions = OrderedDict()
    for i in range(len(markers)-1):
        conditions[markers[i]] = [i+1]
    
    clfs = OrderedDict()

    clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
    clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
    clfs['Xdawn + RegLDA'] = make_pipeline(Xdawn(2, classes=[1]), Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
    clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
    clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())
    methods_list = ['Vect + LR','Vect + RegLDA','Xdawn + RegLDA','ERPCov + TS','ERPCov + MDM']
    # format data
    epochs.pick_types(eeg=True)
    X = epochs.get_data() * 1e6
    times = epochs.times
    y = epochs.events[:, -1]

    # define cross validation 
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=42)

    # run cross validation for each pipeline
    auc = []
    methods = []
    print('Calcul in progress...')
    for m in clfs:
        try: 
            res = cross_val_score(clfs[m], X, y==2, scoring='roc_auc', cv=cv, n_jobs=-1)
            auc.extend(res)
            methods.extend([m]*len(res))
        except:
            pass

    results = pd.DataFrame(data=auc, columns=['AUC'])
    results['Method'] = methods
    
    n_row,n_column = results.shape
    auc_means = []
    for method in methods_list:
        auc = []
        for i in range(n_row):
            if results.loc[i,'Method']== method:
                auc.append(results.loc[i,'AUC'])
        auc_means.append(np.mean(auc))
    counter = 0
    for i in range(len(methods_list)):
        color = 'green' if auc_means[i]>=0.7 else 'red'
        counter = counter +1 if auc_means[i]>=0.7 else counter
       
    return counter > 0, counter