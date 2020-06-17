#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import os.path
from os import listdir
from os.path import isfile, join
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import openpyxl
from tqdm import tqdm

def open_syn_data(syn_name, GENERAL_DIR):
    syn_file = GENERAL_DIR + "\\{}\\{}_Database.csv".format(syn_name, syn_name)
    assert os.path.exists(syn_file), "This path doesn't exist: {}".format(syn_file)

    df_syn = pd.read_csv(syn_file, sep =';')
    df_syn = df_syn[['Patient', 'Age on photo', 'Gender']]
    df_syn.rename(columns={'Patient':'image','Age on photo':'age', 'Gender':'gender'},inplace=True)
    df_syn['gender'] = df_syn['gender'].apply(lambda x: x.lower())
    
    return df_syn


# Make a histogram of all ages
def make_hist(df_syn):
    ages_syn = df_syn.age.values
    plt.hist(ages_syn)
    plt.xlabel("Age")
    plt.title("Syndromic patients patients")
    plt.show()


# open ID control excel sheet
def open_control_excel(control, GENERAL_DIR):
    ID_file = GENERAL_DIR + "\\ID-controls\\all_ID_controls_info_complete.csv"
    assert os.path.exists(ID_file), "This path doesn't exist: {}".format(ID_file)

    df_ID = pd.read_csv(ID_file, sep=';')
    df_ID = df_ID[['pnummer', 'frontal face image', 'agecorrected', 'gender']]
    df_ID = df_ID[df_ID['frontal face image'].notnull()]
    df_ID = df_ID.rename(columns={"frontal face image": "image", "agecorrected": "age"})

    return df_ID

def get_list_age_dif(age_syn):
    if age_syn < 6:
        return [0, 1, -1]
    
    age_dif = int(float(age_syn)/3.0)
    a = list(range(-age_dif, 0))
    a.reverse()
    b = list(range(1, age_dif))
    c = list(zip(b, a))
    c = np.array(c).flatten().tolist()
    return c
    
    

def select_controls(df_syn, df_ID):
    # empty object
    df_select_syn = pd.DataFrame(columns=['image', 'age', 'gender'])
    df_select_ID = pd.DataFrame(columns=['image', 'age', 'gender'])
    
    print("there are {} patients".format(df_syn.shape))
    # find control ID for each syndromic patients
    for index, row in df_syn.iterrows():
        if isinstance(row['age'], int) : 
            age_syn = row['age']
            gender_syn = row['gender']
                           
            # find a control ID with exact same age
            matches_ID = df_ID.loc[(df_ID['age'] == age_syn) & (df_ID['gender'] == gender_syn)]

            # try different age differences
            age_dif = get_list_age_dif(age_syn)
            i = 0   
            while matches_ID.shape[0] == 0:

                matches_ID = df_ID.loc[(df_ID['age'] == age_syn + age_dif[i]) & (df_ID['gender'] == gender_syn)]
                i+= 1
                if i == len(age_dif):
                    break

            if(matches_ID.shape[0] ==0):
                print("For patient {}, gender: {}, age: {}".format(row['image'], row['gender'], row['age']))
                print("No match found within {} and {} years".format(max(age_dif), min(age_dif)))
                continue

            # a match is found, so append sy patient
            df_select_syn = df_select_syn.append({'image': str(row.image) + ".jpg", 'age':row.age, 'gender':row.gender}, ignore_index=True) 

            # pick a random control from this list to append to selected controls
            random_index = random.randint(0, matches_ID.shape[0]-1)
            select_ID = matches_ID.iloc[random_index]
            df_select_ID = df_select_ID.append({'image':str(select_ID.image) + ".jpg", 'age': select_ID.age, 'gender': select_ID.gender}, ignore_index=True)               

            # remove selected row from set of all controls 
            i = df_ID[(df_ID.image == select_ID.image)].index

            OG_shape = df_ID.shape
            df_ID = df_ID.drop(i)
            new_shape = df_ID.shape  

            if(OG_shape[0] - new_shape[0]> 1):
                print("Error")

    print("Done finding all ID controls.")
    return df_select_syn, df_select_ID


def save_info(syn_name, control, df_select_syn, df_select_ID, GENERAL_DIR):
    syn_info_save = GENERAL_DIR + "\\{}-{}\\{}_patients_info.xlsx".format(syn_name, control, syn_name)
    ID_info_save = GENERAL_DIR + "\\{}-{}\\{}_matched_{}_controls_info.xlsx".format(syn_name, control, syn_name, control)
    df_select_syn.to_excel(syn_info_save)
    df_select_ID.to_excel(ID_info_save)


def empty_dir(directory):
    files = [join(directory, f) for f in listdir(directory)]

    for file in files:
        os.remove(file)


### Open Excel files and write the found images to a new directory

def save_img_from_excel_controls(syn_name, control, GENERAL_DIR):
    ID_dir = GENERAL_DIR + "\\{}\\{}-all-photos".format(control, control)
    select_ID_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn_name, control, syn_name, control)
    empty_dir(select_ID_dir)
    
    ID_info_save = GENERAL_DIR + "\\{}-{}\\{}_matched_{}_controls_info.xlsx".format(syn_name, control, syn_name, control)
    df_ID = pd.read_excel(ID_info_save)

    for index,rows in df_ID.iterrows():
        image = str(rows['image'])

        files = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f)) & (image in f))]
        if(len(files)==1):
            im = Image.open(join(ID_dir, files[0]))
            im.save(join(select_ID_dir, files[0]))
        else: 
            print("Manually find image for  {}".format(image))  
            print("in " + str(ID_dir))

def save_img_from_excel_patients(syn_name, control, GENERAL_DIR):
    
    syn_dir = GENERAL_DIR + "\\{}\\{}-all-photos".format(syn_name, syn_name)
    select_syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn_name, control, syn_name)
    empty_dir(select_syn_dir)    
    
    syn_info_save = GENERAL_DIR + "\\{}-{}\\{}_patients_info.xlsx".format(syn_name, control, syn_name)
    df_syn = pd.read_excel(syn_info_save)

    for index,rows in df_syn.iterrows():
        image = rows['image']
        files = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)) and image in f)]
        if(len(files)==1):
            im = Image.open(join(syn_dir, files[0]))
            im.save(join(select_syn_dir, files[0]))
        else: 
            print("Manually find image for image: {}".format(image))

## Write syndrome files and control files to txt 

def save_control_patients_info(syn_name, control, trial_nr, GENERAL_DIR):    
    control_dir = GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn_name, control, syn_name, control)
    control_files = [f for f in listdir(join(control_dir)) if isfile(join(control_dir, f)) and ".jpg" in f ]
   
    syn_dir = GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn_name, control, syn_name)
    syn_files = [f for f in listdir(join(syn_dir)) if isfile(join(syn_dir, f)) and ".jpg" in f ]
   
    control_patient_info = open("results/{}-{}/{}-patient-{}-control-info-run-{}.txt".format(syn_name, control, syn_name, control, trial_nr), "w")
    
    control_patient_info.write("Patients for syndrome {}\n".format(syn_name))
    for syn in syn_files:
        control_patient_info.write(syn + "\n")
   
    control_patient_info.write("\n{} controls for syndrome {}\n".format(control, syn_name))
    for control in control_files:
        control_patient_info.write(control + "\n")
    control_patient_info.close()

def make_dirs(GENERAL_DIR, syn_name, control):

    if not os.path.exists(GENERAL_DIR + "\\{}-{}".format(syn_name, control)):
        os.mkdir(GENERAL_DIR + "\\{}-{}".format(syn_name, control))
            
    if not os.path.exists(GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn_name, control, syn_name, control)):
        os.mkdir(GENERAL_DIR + "\\{}-{}\\{}-selected-{}-controls".format(syn_name, control, syn_name, control))

    if not os.path.exists(GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn_name, control, syn_name)):
        os.mkdir(GENERAL_DIR + "\\{}-{}\\{}-patients".format(syn_name, control, syn_name))
    
    if not os.path.exists(GENERAL_DIR + "\\{}-{}\\representations".format(syn_name, control)):
        os.mkdir(GENERAL_DIR + "\\{}-{}\\representations".format(syn_name, control))
        
    if not os.path.exists("results\\{}-{}".format(syn_name, control)):
        os.mkdir("results\\{}-{}".format(syn_name, control))


def main(GENERAL_DIR, syn_name, control, trial_nr):
    
    make_dirs(GENERAL_DIR, syn_name, control)
        
    print("Selecting controls for trial {} \nfor syndrom: {} and control: {}".format(trial_nr, syn_name, control))
    
    df_syn = open_syn_data(syn_name, GENERAL_DIR)
    df_ID = open_syn_data(control, GENERAL_DIR)
#     make_hist(df_syn)
    
#     make_hist(df_ID)

    df_select_syn, df_select_ID = select_controls(df_syn, df_ID)
    print("Syndrome {} \nShape of patient data {}, shape of {} control data {}".format(syn_name, df_select_syn.shape, control, df_select_ID.shape))

    save_info(syn_name, control, df_select_syn, df_select_ID, GENERAL_DIR)

    save_img_from_excel_controls(syn_name, control, GENERAL_DIR)
    save_img_from_excel_patients(syn_name, control, GENERAL_DIR)

    save_control_patients_info(syn_name, control, trial_nr, GENERAL_DIR)    
                
    print("Done running ID_control_selection.py")

