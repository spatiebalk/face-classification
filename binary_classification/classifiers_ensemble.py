import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import date
from os.path import join, isfile
from os import listdir
import time
from sklearn.metrics import f1_score
import tensorflow as tf 
from statistics import mode
import pointnet_model


def read_rep(syn, syn_csv, ID_csv, data_dir):
    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn)
    ID_dir = data_dir+ "\\{}-selected-ID-controls".format(syn)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
    
    data_syn, data_ID, labels_syn, labels_ID = [], [], [], []
    
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] in files_syn: 
                rep = list(map(float, row[1:]))
                data_syn.append(rep)
                labels_syn.append(1)

    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] in files_ID:
                rep = list(map(float, row[1:]))
                data_ID.append(rep)
                labels_ID.append(0)

    return np.array(data_syn), np.array(data_ID), np.array(labels_syn), np.array(labels_ID)



def read_rep_oc(syn, syn_csv, ID_csv, data_dir):    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn)
    ID_dir = data_dir+ "\\{}-selected-ID-controls".format(syn)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
    
    data_syn, data_ID = [], []

    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                data_syn.append(row)
                    
    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                data_ID.append(row)
    
    return np.array(data_syn), np.array(data_ID)
                
                
def combine_openface_cfps(syn, data_dir):             
                
    method = "openface"
    syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn, method)
    ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
    data_syn_of, data_ID_of = read_rep_oc(syn, syn_csv, ID_csv, data_dir)
                
    method = "cfps"
    syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn, method)
    ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
    data_syn_cfps, data_ID_cfps = read_rep_oc(syn, syn_csv, ID_csv, data_dir)
       
    data_syn, data_ID = [], []
    labels = []
    indices_syn, indices_ID = [], []
    
    for index, (openface_i, cfps_i) in enumerate(zip(data_syn_of, data_syn_cfps)):
        if openface_i[0] == cfps_i[0]:
            data_syn.append(openface_i[1:].tolist() + cfps_i[1:].tolist())
            labels.append(1)
            indices_syn.append(index)     
        
    for index, (openface_i, cfps_i) in enumerate(zip(data_ID_of, data_ID_cfps)):
        if openface_i[0] == cfps_i[0]:
            data_ID.append(openface_i[1:].tolist() + cfps_i[1:].tolist())
            labels.append(0)
            indices_ID.append(index)
    
    data = data_syn + data_ID

    return np.array(data), np.array(labels)




def read_rep_landmarks(syn_name, syn_csv, ID_csv, data_dir):
    
    # open directories
    syn_dir = data_dir+"\\{}-patients".format(syn_name)
    ID_dir = data_dir+ "\\{}-selected-ID-controls".format(syn_name)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f))) and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]

    data_syn, data_ID, labels_syn, labels_ID = [], [], [], []
    
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] in files_syn:
                rep = []
                i = 1
                while i < len(row[1:]):
                    rep.append([float(row[i]), float(row[i+1]), float(row[i+2])])
                    i+=3                       
                data_syn.append(rep)
                labels_syn.append(1)

    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for index, row in enumerate(reader):
            if row[0] in files_ID:
                rep = []
                i = 1
                while i < len(row[1:]):
                    rep.append([float(row[i]), float(row[i+1]), float(row[i+2])])
                    i+=3                       
                data_ID.append(rep)
                labels_ID.append(0)

    return np.array(data_syn), np.array(data_ID), np.array(labels_syn), np.array(labels_ID)



def normalize(data_1, data_2, data_3, data_4):
    data_1 = Normalizer().fit_transform(data_1)
    data_2 = Normalizer().fit_transform(data_2)
    data_3 = Normalizer().fit_transform(data_3)
    data_4 = Normalizer().fit_transform(data_4)

    return data_1, data_2, data_3, data_4


def load_data(syn, GENERAL_DIR, data_dir): 

    method = "deepface"
    syn_csv = data_dir+"\\representations\{}-patients-{}.csv".format(syn, method)
    ID_csv  = data_dir+"\\representations\ID-controls-{}.csv".format(method)
    data_syn_df, data_ID_df, labels_syn_df, labels_ID_df = read_rep(syn, syn_csv, ID_csv, data_dir)
    
    method = "facereader-landmarks"
    syn_csv = GENERAL_DIR+ "\\features_facereader_landmarks_patient_groups.csv"
    ID_csv = GENERAL_DIR+ "\\features_facereader_landmarks_all_controls.csv" 
    data_syn_fr, data_ID_fr, _, _ = read_rep_landmarks(syn, syn_csv, ID_csv, data_dir)    
    
    method = "facereader-landmarks-distances"
    syn_csv = GENERAL_DIR+ "\\features_facereader_landmarks_distances_patient_groups_left_right.csv"
    ID_csv = GENERAL_DIR+ "\\features_facereader_landmarks_distances_all_controls_left_right.csv"    
    data_syn_dis, data_ID_dis, _,  _ = read_rep(syn, syn_csv, ID_csv, data_dir)  
    
    # openface + cfps
    data_oc, labels_oc = combine_openface_cfps(syn, data_dir)
    
    indices_to_keep = []
    
    for index, rep in enumerate(data_syn_dis):
        if not all(v == 0 for v in data_syn_dis[index]) and not all(v == 0 for v in data_ID_dis[index]):
            indices_to_keep.append(index)
                     
    # all deepface data
    data_df = data_syn_df.tolist() + data_ID_df.tolist()
    labels_df = labels_syn_df.tolist() + labels_ID_df.tolist()
    
    # only deepface (that also has a facereader rep)
    data_syn_df_drop = data_syn_df[indices_to_keep]
    data_ID_df_drop = data_ID_df[indices_to_keep]
    data_df_drop = data_syn_df_drop.tolist() + data_ID_df_drop.tolist()
    
    # facereader landmarks 
    data_syn_fr = data_syn_fr[indices_to_keep]
    data_ID_fr = data_ID_fr[indices_to_keep]
    data_fr = data_syn_fr.tolist() + data_ID_fr.tolist()
    
    # only distance with facereader rep
    data_syn_dis = data_syn_dis[indices_to_keep]
    data_ID_dis = data_ID_dis[indices_to_keep]
    data_dis = data_syn_dis.tolist() + data_ID_dis.tolist()
    
    # labels with facereader rep
    labels_syn_df = labels_syn_df[indices_to_keep]
    labels_ID_df = labels_ID_df[indices_to_keep]
    labels = labels_syn_df.tolist() + labels_ID_df.tolist() 

    return np.array(data_df), np.array(data_df_drop), np.array(data_fr), np.array(data_dis), np.array(data_oc).astype(np.float32), np.array(labels_df), np.array(labels), np.array(labels_oc)


def knn_classifier(data, labels):
    y_true, y_probs, y_preds = [], [], [] 
    loo = LeaveOneOut()

    # leave one out split and make prediction
    for train_index, test_index in loo.split(data):
        y_true.append(labels[test_index])
        
        X_train, X_test = np.array(data[train_index]), data[test_index]
        y_train, _ = np.array(labels[train_index]), labels[test_index]
        
        model = KNeighborsClassifier(n_neighbors=3, weights='distance')
        model.fit(X_train, y_train)

        y_probs.append(model.predict_proba(X_test.reshape(1, -1))[:,1])
        y_preds.append(model.predict(X_test.reshape(1, -1)))
        
        del model

    # based on all predictions make aroc curve and confusion matrix
    aroc = roc_auc_score(y_true, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    spec = tn / (tn+fp)  
    sens = tp / (tp+fn)
    f1 = f1_score(y_true, y_preds)
                
    return aroc, spec, sens, f1, y_true, y_probs, y_preds


def pointnet_classifier(data, labels):
    y_true, y_probs, y_preds = [], [], [] 
    loo = LeaveOneOut()
    
    for train_index, test_index in loo.split(data):
        y_true.append(labels[test_index])

        X_train, X_test = np.array(data[train_index]), data[test_index]
        y_train, _ = np.array(labels[train_index]), labels[test_index]

        model = pointnet_model.generate_model()
        model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=4, shuffle=True)

        y_pred_array = model.predict(X_test)
        y_pred = tf.math.argmax(y_pred_array, -1).numpy()

        y_probs.append(y_pred_array[0][1])
        y_preds.append(y_pred) 
        
        del model
        
    aroc = roc_auc_score(y_true, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    spec = tn / (tn+fp)  
    sens = tp / (tp+fn)
    f1 = f1_score(y_true, y_preds)
    
    return aroc, spec, sens, f1, y_true, y_probs, y_preds


def randomforest_classifier(data, labels):
    y_true, y_probs, y_preds = [], [], [] 
    loo = LeaveOneOut()
    
    for train_index, test_index in loo.split(data):
        y_true.append(labels[test_index])

        X_train, X_test = np.array(data[train_index]), data[test_index]
        y_train, _ = np.array(labels[train_index]), labels[test_index]

        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        y_probs.append(model.predict_proba(X_test.reshape(1, -1))[:,1])
        y_preds.append(model.predict(X_test.reshape(1, -1)))
        
        del model

    aroc = roc_auc_score(y_true, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    spec = tn / (tn+fp)  
    sens = tp / (tp+fn)
    f1 = f1_score(y_true, y_preds)
    
    return aroc, spec, sens, f1, y_true, y_probs, y_preds


BATCH_SIZE = 8
def main(GENERAL_DIR, syn_list, trial):

    ## open file 
    results_file = open("results/ensemble_results_3_models_run_{}.txt".format(trial), "w")

    # read in all data (per syndrome) which has a facereader and deepface representation
    for syn in syn_list:

        data_dir = GENERAL_DIR + "\\{}".format(syn) 
        print("Syndrome that will be classified: {} \n\n".format(syn))
        results_file.write("Syndrome: {}\n".format(syn))

        data_df_all, data_df, data_fr, data_dis, data_oc, labels_df_all, labels, labels_oc = load_data(syn, GENERAL_DIR, data_dir)
        data_df_all, data_df, data_dis, data_oc = normalize(data_df_all, data_df, data_dis, data_oc)   

        # DEEPFACE - KNN - all
        aroc_df_all, spec_df_all, sens_df_all, f1_df_all, _, _, _ = knn_classifier(data_df_all, labels_df_all)
        results_file.write("Deepface with {} patients and {} controls\n".format(labels_df_all.tolist().count(1), labels_df_all.tolist().count(0)))
        results_file.write("AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}, F1: {:.4f}\n\n".format(aroc_df_all, spec_df_all, sens_df_all, f1_df_all))

        # DEEPFACE - KNN 
        aroc_df, spec_df, sens_df, f1_df, y_true_df, y_probs_df, y_preds_df = knn_classifier(data_df, labels)
        results_file.write("Deepface with {} patients and {} controls\n".format(labels.tolist().count(1), labels.tolist().count(0)))
        results_file.write("AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}, F1: {:.4f}\n\n".format(aroc_df, spec_df, sens_df, f1_df))

        # POINTNET 
        aroc_pn, spec_pn, sens_pn, f1_pn, y_true_pn, y_probs_pn, y_preds_pn = pointnet_classifier(data_fr, labels)
        results_file.write("Pointnet with {} patients and {} controls\n".format(labels.tolist().count(1), labels.tolist().count(0)))
        results_file.write("AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}, F1: {:.4f}\n\n".format(aroc_pn, spec_pn, sens_pn, f1_pn))

        # RANDOM FOREST 
        aroc_rf, spec_rf, sens_rf, f1_rf, y_true_rf, y_probs_rf, y_preds_rf = randomforest_classifier(data_dis, labels)
        results_file.write("Random Forest with {} patients and {} controls\n".format(labels.tolist().count(1), labels.tolist().count(0)))
        results_file.write("AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}, F1: {:.4f}\n\n".format(aroc_rf, spec_rf, sens_rf, f1_rf))

        # OPENFACE-CFPS
        aroc_oc, spec_oc, sens_oc, f1_oc, _, _, _ = knn_classifier(data_oc, labels_oc)
        results_file.write("Openface-CFPS with {} patients and {} controls\n".format(labels.tolist().count(1), labels.tolist().count(0)))
        results_file.write("AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}, F1: {:.4f}\n\n".format(aroc_oc, spec_oc, sens_oc, f1_oc))

        #ensemble mean 
        ensemble_probs, ensemble_preds = [], []
        for index, _ in enumerate(y_true_df):
            mean_prob = np.mean([y_probs_df[index], y_probs_pn[index], y_probs_rf[index]])
            ensemble_probs.append(mean_prob)

            mode_pred = mode([y_preds_df[index][0], y_preds_pn[index][0], y_preds_rf[index][0]])
            ensemble_preds.append(mode_pred)

        aroc_ensemble = roc_auc_score(y_true_df, ensemble_probs)
        tn_en, fp_en, fn_en, tp_en = confusion_matrix(y_true_df, ensemble_preds).ravel()
        spec_ensemble = tn_en / (tn_en+fp_en)  
        sens_ensemble = tp_en / (tp_en+fn_en)
        f1_ensemble = f1_score(y_true_df, ensemble_preds)

        results_file.write("Ensemble (deepface/pointnet/random_forest) classifier mean and mode \n".format(labels.tolist().count(1), labels.tolist().count(0)))
        results_file.write("AROC: {:.4f}, spec: {:.4f}, sens: {:.4f}, F1: {:.4f}\n\n".format(aroc_ensemble, spec_ensemble, sens_ensemble, f1_ensemble))

    results_file.close()


