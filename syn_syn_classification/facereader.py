# Representations are taken from:
# GENERAL_DIR\features_facereader_landmarks_distances_patient_groups.csv
# GENERAL_DIR\features_facereader_landmarks_distances_all_controls.csv
# and written to
# GENERAL_DIR\syn-control\representations\{}-patients-facereader.csv
# GENERAL_DIR\syn-control\representations\ID-controls-patients-facereader.csv
    
    
import pandas
import csv 
from os.path import join, isfile
from os import listdir
import pandas as pd
import xlrd
import numpy as np

#currently unused
def facereader_reps(GENERAL_DIR, syn, control):
    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn, control, syn)
    ID_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn, control, syn, control)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
    
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))    
    
    syn_csv = GENERAL_DIR + "\\features_facereader_patient_groups.csv"    
       
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                syn_rep.append(row) 
                   
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                ID_rep.append(row)                    

    print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_file_syn = GENERAL_DIR+ "\\{}-{}\\representations\{}-patients-facereader.csv".format(syn, control, syn)
    csv_file_ID = GENERAL_DIR+ "\\{}-{}\\representations\{}-controls-facereader.csv".format(syn, control, control)

    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)

    print("Done with saving all facereader representations for {}-{}.".format(syn_name, control))


def facereader_landmarks_reps(GENERAL_DIR, syn, control):
    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn, control, syn)
    ID_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn, control, syn, control)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
    
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))   
    
    syn_csv = GENERAL_DIR + "\\features_facereader_landmarks_patient_groups.csv"         
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                syn_rep.append(row) 
                   
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                ID_rep.append(row)                    

    print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_file_syn = GENERAL_DIR+ "\\{}-{}\\representations\{}-patients-facereader-landmarks.csv".format(syn, control, syn)
    csv_file_ID = GENERAL_DIR+ "\\{}-{}\\representations\{}-controls-facereader-landmarks.csv".format(syn, control, control)

    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)
        
        
def facereader_landmarks_dis_reps(GENERAL_DIR, syn, control):
    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn, control, syn)
    ID_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn, control, syn, control)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn in f]
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
    
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))   
    
    syn_csv = GENERAL_DIR + "\\features_facereader_landmarks_distances_patient_groups_left_right.csv"          
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_syn:
                syn_rep.append(row) 
                   
    with open (syn_csv, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0] in files_ID:
                ID_rep.append(row)                    

    print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_file_syn = GENERAL_DIR+ "\\{}-{}\\representations\{}-patients-facereader-landmarks-distances.csv".format(syn, control, syn)
    csv_file_ID = GENERAL_DIR+ "\\{}-{}\\representations\{}-controls-facereader-landmarks-distances.csv".format(syn, control, control)

    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)