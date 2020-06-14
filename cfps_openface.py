#!/usr/bin/env python
# coding: utf-8
import pandas
import csv 
from os.path import join, isfile
from os import listdir
import pandas as pd
import xlrd
import numpy as np

def openface_cfps_reps(GENERAL_DIR, method, syn_name):

    syn_rep, ID_rep = [], []

    # open directories
    syn_dir = GENERAL_DIR + "\{}\{}-patients".format(syn_name, syn_name)
    ID_dir = GENERAL_DIR + "\{}\{}-selected-ID-controls".format(syn_name, syn_name)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn_name in f] # "kdv" for KDVS
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".JPG" in f or ".jpg" in f]

    syn_xlsx = GENERAL_DIR + "\\features_"+method+"_patient_groups.xlsx"    
    df_reps = pd.read_excel(syn_xlsx)

    ID_xlsx = GENERAL_DIR + "\\features_"+method+"_all_controls.xlsx"
    df_ID = pd.read_excel(ID_xlsx)

    for (index_rep, row_rep), (index_ID, row_ID) in zip(df_reps.iterrows(), df_ID.iterrows()):

        filename = row_rep.iloc[0]
        # check if row is about kdv patient

        if syn_name in filename:
        #if "kdv" in filename: # kdv specific

            # check if that kdv patient is in the current list
            filename = filename + ".jpg"
            #filename = filename.replace("small_","") # kdv specific
            if filename in files_syn:
                rep_syn = row_rep.iloc[5:].tolist()
                syn_rep.append([filename] + rep_syn)

                rep_ID = row_ID.iloc[5:].tolist()
                ID_rep.append([filename] + rep_ID)    

    print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_file_syn = GENERAL_DIR+ "\\{}\\representations\{}-patients-{}.csv".format(syn_name, syn_name, method)
    csv_file_ID = GENERAL_DIR+ "\\{}\\representations\ID-controls-{}.csv".format(syn_name, method)

    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)

    print("Done with saving all {} representations for {}.".format(method, syn_name))

