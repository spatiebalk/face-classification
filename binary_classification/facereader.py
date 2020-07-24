# Representations are taken from:
# GENERAL_DIR\features_facereader_landmarks_distances_patient_groups.csv
# GENERAL_DIR\features_facereader_landmarks_distances_all_controls.csv
# and written to
# GENERAL_DIR\syn\representations\{}-patients-facereader.csv
# GENERAL_DIR\syn\representations\ID-controls-patients-facereader.csv
    
    
import pandas
import csv 
from os.path import join, isfile
from os import listdir
import pandas as pd
import xlrd
import numpy as np

# function is unused for now as it takes the 93 feature vector of Facereader
def facereader_reps(GENERAL_DIR, syn_name):
    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\{}\{}-patients".format(syn_name, syn_name)
    ID_dir = GENERAL_DIR + "\{}\{}-selected-ID-controls".format(syn_name, syn_name)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn_name in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))

    syn_csv = GENERAL_DIR + "\\features_facereader_patient_groups.csv"       
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                syn_rep.append(row) 
            
    ID_csv = GENERAL_DIR + "\\features_facereader_all_controls.csv"          
    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                ID_rep.append(row)                    

    print("Syn_reps: {}, ID_reps: {}\n".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_syn_selected = GENERAL_DIR+ "\\{}\\representations\{}-patients-facereader.csv".format(syn_name, syn_name)
    csv_ID_selected = GENERAL_DIR+ "\\{}\\representations\ID-controls-facereader.csv".format(syn_name)

    # save representation of kdv patients
    with open(csv_syn_selected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_ID_selected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)

    
# select the relevant reps from the general facereader land mark file to a new file 
def facereader_landmarks_reps(GENERAL_DIR, syn):
    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\{}\{}-patients".format(syn, syn)
    ID_dir = GENERAL_DIR + "\{}\{}-selected-ID-controls".format(syn, syn)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))

    syn_csv = GENERAL_DIR + "\\features_facereader_landmarks_patient_groups.csv"      
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                syn_rep.append(row) 
            
    ID_csv = GENERAL_DIR + "\\features_facereader_landmarks_all_controls.csv"        
    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                ID_rep.append(row)                    

    print("Syn_reps: {}, ID_reps: {}\n".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_syn_selected = GENERAL_DIR+ "\\{}\\representations\{}-patients-facereader-landmarks.csv".format(syn, syn)
    csv_ID_selected = GENERAL_DIR+ "\\{}\\representations\ID-controls-facereader-landmarks.csv".format(syn)

    # save representation of kdv patients
    with open(csv_syn_selected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_ID_selected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)
        

# select the relevant reps from the general facereader landmark distances file to a new file 
def facereader_landmarks_dis_reps(GENERAL_DIR, syn):
    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\{}\{}-patients".format(syn, syn)
    ID_dir = GENERAL_DIR + "\{}\{}-selected-ID-controls".format(syn, syn)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and ".jpg" in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]  
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))

    syn_csv = GENERAL_DIR + "\\features_facereader_landmarks_distances_patient_groups.csv"      
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                syn_rep.append(row) 
            
    ID_csv = GENERAL_DIR + "\\features_facereader_landmarks_distances_all_controls.csv"        
    with open (ID_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                ID_rep.append(row)                    

    print("Syn_reps: {}, ID_reps: {}\n".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_syn_selected = GENERAL_DIR+ "\\{}\\representations\{}-patients-facereader-landmarks-distances.csv".format(syn, syn)
    csv_ID_selected = GENERAL_DIR+ "\\{}\\representations\ID-controls-facereader-landmarks-distances.csv".format(syn)

    # save representation of kdv patients
    with open(csv_syn_selected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_ID_selected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)
        