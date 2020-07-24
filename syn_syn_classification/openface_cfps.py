# Representations are taken from:
# GENERAL_DIR\features_(openface\cfps)_patient_groups.xlsx   
# GENERAL_DIR\features_(openface\cfps)_all_controls.xlsx   
# and written to
# GENERAL_DIR\syn\representations\syn-control\syn-patients-(openface\cfps).csv
# GENERAL_DIR\syn\representations\syn-control\syn-selected-ID-control(openface\cfps).csv

import pandas
import csv 
from os.path import join, isfile
from os import listdir
import pandas as pd
import xlrd
import numpy as np

def openface_cfps_reps(GENERAL_DIR, syn, control):

    # for each method
    for method in ["openface", "cfps"]:
        syn_rep, ID_rep = [], []

        # open directories
        syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn, control, syn)
        ID_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn, control, syn, control)

        # get list of filenames
        files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn in f] 
        files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
        print("{}: syn_files: {}, ID_files: {}".format(method, len(files_syn), len(files_ID)))

        syn_xlsx = GENERAL_DIR + "\\features_"+method+"_patient_groups.xlsx"    
        df_reps = pd.read_excel(syn_xlsx)

        for index_rep, row_rep  in df_reps.iterrows():
            filename = row_rep.iloc[0]
            if filename +".jpg" in files_syn: # openface is saved without extension
                rep_syn = row_rep.iloc[5:].tolist()
                syn_rep.append([filename] + rep_syn)

        for index_ID, row_ID in  df_reps.iterrows():
            filename = row_ID.iloc[0]
            if filename +".jpg" in files_ID:
                rep_ID = row_ID.iloc[5:].tolist()
                ID_rep.append([filename] + rep_ID)

        print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))

        # location to save representation
        csv_file_syn = GENERAL_DIR+ "\\{}-{}\\representations\{}-patients-{}.csv".format(syn, control, syn, method)
        csv_file_ID = GENERAL_DIR+ "\\{}-{}\\representations\{}-controls-{}.csv".format(syn, control, control, method)

        # save representation of kdv patients
        with open(csv_file_syn, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(syn_rep)

        # save representation of ID controls
        with open(csv_file_ID, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(ID_rep)


