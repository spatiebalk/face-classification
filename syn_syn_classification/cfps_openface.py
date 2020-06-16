#!/usr/bin/env python
# coding: utf-8
import pandas
import csv 
from os.path import join, isfile
from os import listdir
import pandas as pd
import xlrd
import numpy as np

def openface_cfps_reps(GENERAL_DIR, method, syn_name, control):

    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn_name, control, syn_name)
    ID_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn_name, control, syn_name, control)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn_name in f] 
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and control in f]
    
    print("Syn_files: {}, ID_files: {}".format(len(files_syn), len(files_ID)))
    
    syn_xlsx = GENERAL_DIR + "\\features_"+method+"_patient_groups.xlsx"    
    df_reps = pd.read_excel(syn_xlsx)

    #ID_xlsx = GENERAL_DIR + "\\features_"+method+"_all_controls.xlsx"
    #df_ID = pd.read_excel(ID_xlsx)

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
    csv_file_syn = GENERAL_DIR+ "\\{}-{}\\representations\{}-patients-{}.csv".format(syn_name, control, syn_name, method)
    csv_file_ID = GENERAL_DIR+ "\\{}-{}\\representations\{}-controls-{}.csv".format(syn_name, control, control, method)

    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)

    print("Done with saving all {} representations for {}-{}.".format(method, syn_name, control))

