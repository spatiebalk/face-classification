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


def read_rep(syn_name, syn_csv, ID_csv, data_dir):
    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn_name)
    ID_dir = data_dir+ "\\{}-selected-ID-controls".format(syn_name)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
    
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


def read_rep2(syn_name, syn_csv, ID_csv, data_dir):
    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn_name)
    ID_dir = data_dir+ "\\{}-selected-ID-controls".format(syn_name)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
        
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


def plot_pca_tsne(data, labels, lowest_age = -1, highest_age = -1):
    plt.figure(figsize=(12,6))
    plt.plot([1,2])

    # visualize data in tnse (men/women)
    X_embedded_tsne = TSNE(n_components=2, init='pca').fit_transform(data)

    plt.subplot(121)
    unique = list(set(labels))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [X_embedded_tsne[j, 0] for j  in range(len(X_embedded_tsne[:,0])) if labels[j] == u]
        yi = [X_embedded_tsne[j, 1] for j  in range(len(X_embedded_tsne[:,1])) if labels[j] == u]
        plt.scatter(xi, yi, c=[colors[i]], label=str(u))
    plt.legend()
    plt.title("t-sne for age range {}-{}".format(lowest_age, highest_age))

    # visualize data in pca (men/women)
    X_embedded_pca = PCA(n_components=2).fit_transform(data)

    plt.subplot(122)
    unique = list(set(labels))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [X_embedded_pca[j, 0] for j  in range(len(X_embedded_pca[:,0])) if labels[j] == u]
        yi = [X_embedded_pca[j, 1] for j  in range(len(X_embedded_pca[:,1])) if labels[j] == u]
        plt.scatter(xi, yi, c=[colors[i]], label=str(u))
    plt.legend()
    plt.title("pca for age range{}-{}".format(lowest_age, highest_age))

    plt.show()


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


def knn_classifier(data, labels):
    k_values = [3] #  [3,5,7,9,12]
    best_aroc = 0
    best_k = 0
    best_norm = -1
    best_spec,best_sens = 0, 0

    for k in tqdm(k_values):
        # can't have more neighbors than samples
        if k < data.shape[0]:
            for i in [1]: #[0, 1, 2]:
                data = normalize(data, i) 
                all_y, all_probs, all_preds = [], [], [] 
                loo = LeaveOneOut()

                # leave one out split and make prediction
                for train, test in loo.split(data):
                    all_y.append(labels[test])
                    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
                    model = model.fit(data[train], labels[train])                
                    all_probs.append(model.predict_proba(data[test].reshape(1, -1))[:,1])
                    all_preds.append(model.predict(data[test].reshape(1, -1)))

                # based on all predictions make aroc curve and confusion matrix
                aroc = roc_auc_score(all_y, all_probs)
                tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()
                spec = tn / (tn+fp)  
                sens = tp / (tp+fn)

                if aroc > best_aroc:
                    best_aroc, best_spec, best_sens, best_norm = aroc, spec, sens, i 
                    best_k = k
                
    return best_k, best_norm, best_aroc, best_spec, best_sens


def svm_classifier(data, labels):
    kernels = ['linear'] # , 'poly', 'rbf', 'sigmoid']
    best_aroc = 0
    best_kernel = None
    best_norm = -1
    best_spec,best_sens = 0, 0

    for k in tqdm(kernels):
        for i in [1]: #[0, 1, 2]:
            
            data = normalize(data, i) 
            all_y, all_probs, all_preds = [], [], [] 
            loo = LeaveOneOut()
            
            # leave one out split and make prediction
            for train, test in loo.split(data):
                all_y.append(labels[test])
                model = SVC(kernel=k, probability=True)
                model = model.fit(data[train], labels[train])
                all_probs.append(model.predict_proba(data[test].reshape(1, -1))[:,1])
                all_preds.append(model.predict(data[test].reshape(1, -1)))

            # based on all predictions make aroc curve and confusion matrix
            aroc = roc_auc_score(all_y, all_probs)
            tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()
            spec = tn / (tn+fp)  
            sens = tp / (tp+fn)
               
            if aroc > best_aroc:
                best_aroc, best_spec, best_sens, best_norm = aroc, spec, sens, i 
                best_kernel = k
                
    return best_kernel, best_norm, best_aroc, best_spec, best_sens


def rf_classifier(data, labels):
    best_aroc = 0
    estimators = [10] # [5, 10, 20, 40] 
    best_estimator_rf = 0
    best_norm = -1
    best_spec,best_sens = 0, 0

    for est in tqdm(estimators):
        for i in [1]: #[0, 1, 2]:
            
            data = normalize(data, i) 
            all_y, all_probs, all_preds = [], [], [] 
            loo = LeaveOneOut()
            
            # leave one out split and make prediction
            for train, test in loo.split(data):
                all_y.append(labels[test])
                model = RandomForestClassifier(n_estimators=est)
                model = model.fit(data[train], labels[train])
                all_probs.append(model.predict_proba(data[test].reshape(1, -1))[:,1])
                all_preds.append(model.predict(data[test].reshape(1, -1)))

            # based on all predictions make aroc curve and confusion matrix
            aroc = roc_auc_score(all_y, all_probs)
            tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()
            spec = tn / (tn+fp)  
            sens = tp / (tp+fn)
               
            if aroc > best_aroc:
                best_aroc, best_spec, best_sens, best_norm = aroc, spec, sens, i 
                best_estimator_rf = est
    
    return best_estimator_rf, best_norm, best_aroc, best_spec, best_sens


def gr_classifier(data, labels):
    best_aroc = 0
    estimators = [10] # [5, 10, 20, 40]
    best_estimator_gr = 0
    best_norm = -1
    best_spec,best_sens = 0, 0

    for est in tqdm(estimators):
        for i in [1]: #[0, 1, 2]:
            
            data = normalize(data, i) 
            all_y, all_probs, all_preds = [], [], [] 
            loo = LeaveOneOut()
            
            # leave one out split and make prediction
            for train, test in loo.split(data):
                all_y.append(labels[test])
                model = GradientBoostingClassifier(n_estimators=est)
                model = model.fit(data[train], labels[train])
                all_probs.append(model.predict_proba(data[test].reshape(1, -1))[:,1])
                all_preds.append(model.predict(data[test].reshape(1, -1)))

            # based on all predictions make aroc curve and confusion matrix
            aroc = roc_auc_score(all_y, all_probs)
            tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()
            spec = tn / (tn+fp)  
            sens = tp / (tp+fn)
               
            if aroc > best_aroc:
                best_aroc, best_spec, best_sens, best_norm = aroc, spec, sens, i 
                best_estimator_gr = est
                 
                            
    return best_estimator_gr, best_norm, best_aroc, best_spec, best_sens


def ada_classifier(data, labels):
    best_aroc = 0
    estimators = [10] #[5, 10, 20, 40]
    best_estimator_ada = 0
    best_norm = -1
    best_spec,best_sens = 0, 0

    for est in tqdm(estimators):
        for i in [1]: #[0,1, 2]:
            
            data = normalize(data, i) 
            all_y, all_probs, all_preds = [], [], [] 
            loo = LeaveOneOut()
            
            # leave one out split and make prediction
            for train, test in loo.split(data):
                all_y.append(labels[test])
                model = AdaBoostClassifier(n_estimators=est)
                model = model.fit(data[train], labels[train])
                all_probs.append(model.predict_proba(data[test].reshape(1, -1))[:,1])
                all_preds.append(model.predict(data[test].reshape(1, -1)))

            # based on all predictions make aroc curve and confusion matrix
            aroc = roc_auc_score(all_y, all_probs)
            tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()
            spec = tn / (tn+fp)  
            sens = tp / (tp+fn)
               
            if aroc > best_aroc:
                best_aroc, best_spec, best_sens, best_norm = aroc, spec, sens, i 
                best_estimator_ada = est
                
    return best_estimator_ada, best_norm, best_aroc, best_spec, best_sens


def concatenate(syn_name, data_dir, data_combination, nr_feats): 

    method = "deepface"
    syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn_name, method)
    ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
    data_df, labels_df = read_rep(syn_name, syn_csv, ID_csv, data_dir)
    
    method = "dlib"
    syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn_name, method)
    ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
    data_dlib, labels_dlib = read_rep(syn_name, syn_csv, ID_csv, data_dir)

    
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
        syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn_name, method)
        ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
        data_openface, labels_openface = read_rep2(syn_name, syn_csv, ID_csv, data_dir)
        
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
        syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn_name, method)
        ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
        data_cfps, labels_cfps = read_rep2(syn_name, syn_csv, ID_csv, data_dir)
        
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
        syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn_name, method)  
        ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)

        data_fr, labels_fr = read_rep(syn_name, syn_csv, ID_csv, data_dir)      
        
        data, labels  = [], []
        for index, fr_i in enumerate(data_fr):
            if not all(v == 0 for v in fr_i):
                data.append(fr_i)
                labels.append(labels_fr[index])
    

    if data_combination == 9: # facereader landmarks
        method = "facereader-landmarks-distances"
        syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn_name, method)
        ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
        data_fr, labels_fr = read_rep(syn_name, syn_csv, ID_csv, data_dir)

        data, labels  = [], []
        for index, data_i in enumerate(data_fr):
            if not all(v == 0 for v in data_i):
                #only if a face is found
                data.append(data_i) 
                labels.append(labels_fr[index])
                
    return 0, np.array(data), np.array(labels)
              


def get_header(data_combination, nr_feats):
    if data_combination == 0:
        return "0: Classifying data with deepface representation\n\n"
        
    if data_combination == 1:
        return"1: Classifying data with dlib representation\n\n"
            
    if data_combination == 2:
        return "2: Classifying data with all deepface+dlib representations\n\n"
            
    if data_combination == 3:
        return "3: Classifying data with the {} most important features of deepface-dlib representations\n\n".format(nr_feats)
        
    if data_combination == 4:
        return "4: Classifying data with PCA components of deepface-dlib representation\n"
    
    if data_combination == 5:
        return "5: Classifying data with openface representation\n\n"
    
    if data_combination == 6:
        return "6: Classifying data with cfps representation\n\n"
    
    if data_combination == 7:
        return "7: Classifying data with openface+cfps representation\n\n"
    
    if data_combination == 8:
        return "8: Classifying data with facereader representation\n\n"

    if data_combination == 9:
        return "9: Classifying data with facereader-landmarks-distance representation\n\n"

def main(GENERAL_DIR, syn_list, trial_nr):    
    
    for syn_name in syn_list:
        
        today = date.today()
        start = time.time()

        data_dir = GENERAL_DIR + "\\{}".format(syn_name) 
        results_file = open("results/{}/{}-results-deepface-gr-run-{}-{}.txt".format(syn_name, syn_name, trial_nr, today), "w")
        results_file.write("Syndrome that will be classified: {} \n\n".format(syn_name))
        print("Syndrome that will be classified: {} \n\n".format(syn_name))

        nr_feats = 300

        for data_combination in [0]: #0,1, 2, 7, 8]: #, 4, 5, 6, 7, 8]: 

            results_file.write(get_header(data_combination, nr_feats))
            print(get_header(data_combination, nr_feats))            

            nr_comps, data, labels = concatenate(syn_name, data_dir, data_combination, nr_feats) 
          
            if labels.tolist().count(1) <= 3:
                results_file.write("NO RESULTS as there are {} patients and {} controls with a representation\n\n".format(labels.tolist().count(1), labels.tolist().count(0)))
                continue
            
            print("Data shape: {} and labels shape: {}".format(data.shape, labels.shape))
            print("Amount of negatives: {} and positives: {}".format(labels.tolist().count(0), labels.tolist().count(1)))
            
            # continue # so no classifying
            
            results_file.write("Shape of data: {} patients, {} controls, {} features \n\n".format(labels.tolist().count(1), labels.tolist().count(0), data.shape[1]))                          
            if data_combination == 4:
                results_file.write("Nr of pca components used: {}\n\n".format(nr_comps))

            # plot representation
            # plot_pca_tsne(data, labels, low_age, high_age)

            results_file.write("CLASSIFIER RESULTS for {} patients and controls \n".format(syn_name))

#             k, knn_norm, knn_aroc, knn_spec, knn_sens = knn_classifier(data, labels)
#             results_file.write("knn classifier (k = {}), normalize : {} \n    AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}\n".format(k, knn_norm, knn_aroc, knn_spec, knn_sens))

#             kernel, svm_norm, svm_aroc, svm_spec, svm_sens = svm_classifier(data, labels)
#             results_file.write("svm classifier (k = {}), normalize : {} \n    AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}\n".format(kernel, svm_norm, svm_aroc, svm_spec, svm_sens))

#             n_trees_rf, rf_norm, rf_aroc, rf_spec, rf_sens = rf_classifier(data, labels)
#             results_file.write("Random Forest classifier (trees = {}), normalize : {} \n    AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}\n".format(n_trees_rf, rf_norm, rf_aroc, rf_spec, rf_sens))

            n_trees_gr, gr_norm, gr_aroc, gr_spec, gr_sens = gr_classifier(data, labels)
            results_file.write("Gradient Boost classifier (trees = {}), normalize : {} \n    AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}\n".format(n_trees_gr, gr_norm, gr_aroc, gr_spec, gr_sens))

#             n_trees_ada, ada_norm, ada_aroc, ada_spec, ada_sens = ada_classifier(data, labels)
#             results_file.write("Ada Boost classifier (trees = {}), normalize : {} \n    AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}\n".format(n_trees_ada, ada_norm, ada_aroc, ada_spec, ada_sens))

            results_file.write("\n")

        end = time.time()
        results_file.write("Running this whole file took {:.2f} hours".format((end-start)/3600.00))
        results_file.close()
        
    print("Done running classifiers_general.py")
