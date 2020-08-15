#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import itertools
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from datetime import date
from os.path import join, isfile
from os import listdir
import time
import seaborn as sns


def read_rep(syn_name, control, syn_csv, ID_csv, data_dir):
    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn_name)
    ID_dir = data_dir+ "\\{}-selected-{}-controls".format(syn_name, control)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and syn_name in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
    
    data, labels, indices_to_drop = [], [], []

    data_syn = []
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] in files_syn: 
                rep = list(map(float, row[1:]))
                data_syn.append(rep)
                if all(v == 0 for v in rep):
                    indices_to_drop.append(index)
                    
    data_ID = []                    
    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] in files_ID:
                rep = list(map(float, row[1:]))
                data_ID.append(rep)
                if all(v == 0 for v in rep):
                    indices_to_drop.append(index)
    

    for index, (syn_item, ID_item) in enumerate(zip(data_syn, data_ID)):
        if index not in indices_to_drop:
            data.append(syn_item)
            labels.append(1)
            data.append(ID_item)
            labels.append(0)

    return np.array(data), np.array(labels)


def read_rep2(syn_name, control, syn_csv, ID_csv, data_dir):
    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn_name)
    ID_dir = data_dir+ "\\{}-selected-{}-controls".format(syn_name, control)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and syn_name in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
        
    data, labels, indices_to_drop = [], [], []
   
    data_syn = []
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] +".jpg" in files_syn: # openface is saved without extension
                rep = list(map(float, row[1:]))
                data_syn.append(row)
                if all(v == 0 for v in rep):
                    indices_to_drop.append(index)
            else:
                print("image that couldn't be found: {}".format(row[0]))
                
                    
    data_ID = []                    
    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] + ".jpg" in files_ID:
                rep = list(map(float, row[1:]))
                data_ID.append(row)
                if all(v == 0 for v in rep):
                    indices_to_drop.append(index)
        
    for index, (syn_item, ID_item) in enumerate(zip(data_syn, data_ID)):
        if index not in indices_to_drop:          
            data.append(syn_item)
            labels.append(1)
            data.append(ID_item)
            labels.append(0)         
    return np.array(data), np.array(labels)




def plot_roc_curve(y_true, y_pred): 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure(1, figsize=(12,6))
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, lw=2, alpha=0.5, label='LOOCV ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def normalize(data, i):

    if i == 0:
        return data
    
    if i == 1:
        return Normalizer().fit_transform(data)
        
    if i == 2:
        return StandardScaler().fit_transform(data)


def svm_classifier_conf_matrix(data, labels, kernel, norm):

    data = normalize(data, norm) 
    all_y, all_probs, all_preds = [], [], [] 

    loo = LeaveOneOut()

    # leave one out split and make prediction
    for train, test in loo.split(data):
        all_y.append(labels[test])
        model = SVC(kernel=kernel, probability=True)
        model = model.fit(data[train], labels[train])
        all_probs.append(model.predict_proba(data[test].reshape(1, -1))[:,1])
        all_preds.append(model.predict(data[test].reshape(1, -1)))

    # based on all predictions make aroc curve and confusion matrix
    aroc = roc_auc_score(all_y, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()

    return tn, fp, fn, tp, aroc


def concatenate(syn_name, control, data_dir, data_combination, nr_feats): 

    method = "deepface"
    syn_csv = data_dir+"\\representations\\{}-patients-{}.csv".format(syn_name, method)
    ID_csv  = data_dir+"\\representations\\{}-controls-{}.csv".format(control, method)
    data_df, labels_df = read_rep(syn_name, control, syn_csv, ID_csv, data_dir)
    
    method = "dlib"
    syn_csv = data_dir+"\\representations\\{}-patients-{}.csv".format(syn_name, method)
    ID_csv  = data_dir+"\\representations\\{}-controls-{}.csv".format(control, method)
    data_dlib, labels_dlib = read_rep(syn_name, control, syn_csv, ID_csv, data_dir)

    
    if data_combination == 0: 
        # only deepface
        data = data_df
        labels = labels_df
    
    if data_combination == 1: 
        # only dlib
        data, labels  = [], []
        for index, dlib_i in enumerate(data_dlib):
            if not all(v == 0 for v in dlib_i):
                #only if a face is found
                data.append(dlib_i) 
                labels.append(labels_dlib[index])
                
                
    if data_combination == 2 or data_combination == 3 or data_combination == 4:
        # deepface + dlib (all features) 
        data, labels  = [], []
        for index, (df_i, dlib_i) in enumerate(zip(data_df, data_dlib)):
            if not all(v == 0 for v in dlib_i):
                #only if a face is found 
                if not isinstance(df_i, list):
                    df_i = df_i.tolist()
                if not isinstance(dlib_i, list):
                    dlib_i = dlib_i.tolist()  
                    
                data.append(df_i+dlib_i) # concatenation of 4096 deepface + 2210 dlib
                labels.append(labels_df[index])
                
                                               
    if data_combination == 3:
        # deepface + dlib (x most important features)
        # data, labels are already filled from the above if statement
                                               
        # using a Random Forest the x most important features are used                                   
        forest = RandomForestClassifier(n_estimators=10,random_state=0) # 10 has been found with best aroc scores
        forest.fit(data, labels)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        indices = indices[0:nr_feats] 

        data2 = []
        for row in data:
            data2.append(np.array(row)[indices])                                
        data = data2

                                               
    nr_comps = 0
    if data_combination == 4:
        # pca components that explain > 0.9 variance
        for i in range(0, np.array(data).shape[0]):
            pca = PCA(n_components=i)
            components = pca.fit_transform(data)    
            if sum(pca.explained_variance_ratio_) > 0.9:
                nr_comps = i
        
        pca = PCA(n_components=nr_comps)
        data = pca.fit_transform(data)       
        
    
    if data_combination == 5 or data_combination == 7:
        # openface 
        method = "openface"
        syn_csv = data_dir+"\\representations\\{}-patients-{}.csv".format(syn_name, method)
        ID_csv  = data_dir+"\\representations\\{}-controls-{}.csv".format(control, method)
        data_openface, labels_openface = read_rep2(syn_name, control, syn_csv, ID_csv, data_dir)
        
        data = []
        openface_names = data_openface[:,0]
        data_openface = np.array(data_openface)[:, 1:]
        for openface_i in data_openface:
            rep = [float(i) for i in openface_i.tolist()]
            data.append(rep)

        labels = np.array(labels_openface)
        
        
    if data_combination == 6 or data_combination == 7:
        # cfps        
        method = "cfps"
        syn_csv = data_dir+"\\representations\\{}-patients-{}.csv".format(syn_name, method)
        ID_csv  = data_dir+"\\representations\\{}-controls-{}.csv".format(control, method)
        data_cfps, labels_cfps = read_rep2(syn_name, control, syn_csv, ID_csv, data_dir)
        
        data = []
        cfps_names = data_cfps[:,0]
        data_cfps = np.array(data_cfps)[:, 1:]
        
        for cfps_i in data_cfps:
            rep = [float(i) for i in cfps_i.tolist()]
            data.append(rep)
            
        labels = np.array(labels_cfps)

        
    if data_combination == 7:
        # openface + cfps 
        matches = [i==j for i, j in zip(openface_names, cfps_names)]
        
        data, labels  = [], []
        for index, (openface_i, cfps_i) in enumerate(zip(data_openface, data_cfps)):
            if openface_names[index] in cfps_names:
            #if(matches[index]):
                if not isinstance(openface_i, list):
                    openface_i = openface_i.tolist()
                if not isinstance(cfps_i, list):
                    cfps_i = cfps_i.tolist()  
                    
                rep_list = openface_i+cfps_i
                rep = [float(i) for i in rep_list]
                data.append(rep) # concatenation of 128 openface + 340 cfps
                labels.append(labels_openface[index].astype(np.float64))
                
    
    if data_combination == 8:
        # facereader
        method = "facereader"
        syn_csv = data_dir+"\\representations\\{}-patients-{}.csv".format(syn_name, method)
        ID_csv  = data_dir+"\\representations\\{}-controls-{}.csv".format(control, method)

        data_fr, labels_fr = read_rep(syn_name, control, syn_csv, ID_csv, data_dir)      
        
        data, labels  = [], []
        for index, fr_i in enumerate(data_fr):
            if not all(v == 0 for v in fr_i):
                data.append(fr_i)
                labels.append(labels_fr[index])
    
    return 0, np.array(data), np.array(labels)


def plot_conf_matrix(tn, fp, fn, tp, aroc, syn_name, control, classifier, norm):
    
    spec = tn / (tn+fp)  
    sens = tp / (tp+fn)
   
    
    conf_matrix = [[tp, fp],
             [fn, tn]]
    df_cm = pd.DataFrame(conf_matrix, index = ["{}_pred".format(syn_name), "{}_pred".format(control)],
                      columns = [syn_name, control])
    plt.figure(figsize = (8, 6))
    sns_heat = sns.heatmap(df_cm, annot=True)
    plt.title(" {} with  {} and normalization: {}\n aroc: {:.3f}, spec: {:.3f}, sens: {:.3f}".format(syn_name, classifier, norm, aroc, spec, sens))
    
    #plt.savefig("results/{}/{}_conf_matrix.jpg".format(syn_name, syn_name))
    plt.show()

    

def main(GENERAL_DIR, syn_name, control, trial_nr):    
    
        
    today = date.today()
    start = time.time()

    
    
    # data combination
    data_combination = 2
    
    # classifier
    classifier =  "svm"
    
    # parameters
    params = ['poly']
    
    # normalization
    norm = 1
    
    data_dir = GENERAL_DIR + "\\{}-{}".format(syn_name, control) 

    nr_comps, data, labels = concatenate(syn_name, control, data_dir, data_combination, 0) 
    
    if classifier == "svm":
        tn, fp, fn, tp, aroc  = svm_classifier_conf_matrix(data, labels, params[0], norm)
        plot_conf_matrix(tn, fp, fn, tp, aroc, syn_name, control, classifier, norm)
        
  


        
    print("Done running classifiers_general.py")
