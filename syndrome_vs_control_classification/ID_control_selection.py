# syndrome image data should be in: GENERAL_DIR\syn\syn-all-photos and is save to GENERAL_DIR\syn\syn-patients
# control image data should be in: GENERAL_DIR\ID-controls
# syndrome excel info should be in GENERAL_DIR\syn\syn_Database.csv
# control excel info should be in GENERAL_DIR\ID-controls\all_ID_controls_info_complete.csv"

import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import openpyxl

# open syn csv sheet
def open_syn_excel(GENERAL_DIR, syn):
    syn_file = GENERAL_DIR + "\\{}\\{}_Database.csv".format(syn, syn)
    assert os.path.exists(syn_file), "This path doesn't exist: {}".format(syn_file)

    df_syn = pd.read_csv(syn_file, sep =';')
    df_syn = df_syn[['Patient', 'Age on photo', 'Gender']]
    df_syn.rename(columns={'Patient':'image','Age on photo':'age', 'Gender':'gender'},inplace=True)  
    return df_syn


# open ID control excel sheet
def open_control_excel(GENERAL_DIR, syn):
    ID_file = GENERAL_DIR + "\\ID-controls\\all_ID_controls_info_complete.csv"
    assert os.path.exists(ID_file), "This path doesn't exist: {}".format(ID_file)

    df_ID = pd.read_csv(ID_file, sep=';')
    df_ID = df_ID[['pnummer', 'frontal face image', 'agecorrected', 'gender']]
    df_ID = df_ID[df_ID['frontal face image'].notnull()]
    df_ID = df_ID.rename(columns={"frontal face image": "image", "agecorrected": "age"})
    return df_ID


# # Make a histogram of all ages
# def make_hist(df_syn):
#     ages_syn = df_syn.age.values
#     plt.hist(ages_syn)
#     plt.xlabel("Age")
#     plt.title("Syndromic patients patients")
#     plt.show()
   
    
# get list of possible age differences (1/3 higher or lower)
def get_list_age_dif(age_syn):
    age_dif = int(float(age_syn)/3.0)
    a = list(range(-age_dif, 0))
    a.reverse()
    b = list(range(0, age_dif + 1))
    c = list(zip(b, a))
    return np.array(c).flatten().tolist()
    
    
# for each patient select a suitable control
def select_controls(GENERAL_DIR, syn, df_syn, df_ID):
    df_select_syn = pd.DataFrame(columns=['image', 'age', 'gender'])
    df_select_ID = pd.DataFrame(columns=['image', 'age', 'gender'])
    
    # for each patient, find a control
    for index, row in df_syn.iterrows():
        if isinstance(row['age'], int) : 
            age_syn = row['age']
            gender_syn = row['gender'].lower()

            # find a control ID with exact same age
            matches_ID = df_ID.loc[(df_ID['age'] == age_syn) & (df_ID['gender'] == gender_syn)]

            # get possible age difference list
            age_dif = get_list_age_dif(age_syn)

            # if no control is found with the same age, try the age difference list
            for dif in age_dif:
                matches_ID = df_ID.loc[(df_ID['age'] == age_syn + dif) & (df_ID['gender'] == gender_syn)]
                # stop loop if at least one control is found
                if matches_ID.shape[0] != 0:
                    break
            
            # if no control can be found, skip this patient
            if(matches_ID.shape[0] == 0):
                print("For patient {}, gender: {}, age: {}".format(row['image'], row['gender'], row['age']))
                print("No match found within {} and {} years".format(max(age_dif), min(age_dif)))
                
                # save to unselected patient   
                directory = GENERAL_DIR + "\\{}\\{}-unselected".format(syn, syn)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                # open image
                img = cv2.imread(GENERAL_DIR + "\\{}\\{}-patients\\{}.jpg".format(syn, syn, row['image']))
                                 
                # save it to new dir
                cv2.imwrite(directory + "\\{}.jpg".format(row['image']), img)
                continue

            # control is found, so add patient to dataframe
            df_select_syn = df_select_syn.append({'image': str(row.image) + ".jpg", 'age':row.age, 'gender':row.gender}, ignore_index=True) 

            # pick a random control from this list to append to selected controls
            random_index = random.randint(0, matches_ID.shape[0]-1)
            select_ID = matches_ID.iloc[random_index]
            df_select_ID = df_select_ID.append({'image':str(select_ID.pnummer) + '_small_'+ str(select_ID.image), 'age': select_ID.age, 'gender': select_ID.gender}, ignore_index=True)               

            # remove this one control from all controls
            i = df_ID[(df_ID.image == select_ID.image) & (df_ID.pnummer == select_ID.pnummer)].index

            original_shape = df_ID.shape
            df_ID = df_ID.drop(i)
            new_shape = df_ID.shape  
            
            assert original_shape[0] - new_shape[0] == 1, "More than one control removed from control Dataframe"   
                                 
    return df_select_syn, df_select_ID

                         
# save the info about the selected patients in two excel files
def save_info_to_excel(GENERAL_DIR, syn, df_select_syn, df_select_ID):
    syn_info_save = GENERAL_DIR + "\\{}\\{}_patients_info.xlsx".format(syn, syn)
    ID_info_save = GENERAL_DIR + "\\{}\\{}_matched_ID_controls_info.xlsx".format(syn, syn)
    df_select_syn.to_excel(syn_info_save)
    df_select_ID.to_excel(ID_info_save)

                         
# empty directory, before writing images to it
def empty_dir(directory):
    files = [join(directory, f) for f in listdir(directory)]
    for file in files:
        os.remove(file)


# open excel files and write selected control images to a new directory
def save_img_from_excel_controls(GENERAL_DIR, syn):
    ID_dir = GENERAL_DIR + "\\ID-controls"
    select_ID_dir = GENERAL_DIR + "\\{}\\{}-selected-ID-controls".format(syn, syn)
    empty_dir(select_ID_dir) 
    
    ID_info_save = GENERAL_DIR + "\\{}\\{}_matched_ID_controls_info.xlsx".format(syn, syn)
    df_ID = pd.read_excel(ID_info_save)

    for index,rows in df_ID.iterrows():
        image = str(rows['image'])

        files = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f)) & (image in f))]
        if(len(files)==1):
            img = cv2.imread(join(ID_dir, files[0]))
            cv2.imwrite(join(select_ID_dir, files[0]), img)
        else: 
            print("Manually find image for {} in {}".format(image, ID_dir))  


# open excel files and write selected patient images to a new directory
def save_img_from_excel_patients(GENERAL_DIR, syn):
    syn_dir = GENERAL_DIR + "\\{}\\{}-all-photos".format(syn, syn)
    select_syn_dir = GENERAL_DIR + "\\{}\\{}-patients".format(syn, syn)
    empty_dir(select_syn_dir)    
    
    syn_info_save = GENERAL_DIR + "\\{}\\{}_patients_info.xlsx".format(syn, syn)
    df_syn = pd.read_excel(syn_info_save)

    for index,rows in df_syn.iterrows():
        image = rows['image']
        files = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)) and image in f)]
        if(len(files)==1):
            img = cv2.imread(join(syn_dir, files[0]))
            cv2.imwrite(join(select_syn_dir, files[0]), img)
        else: 
            print("Manually find image for {} in {}".format(image, syn_dir))

                         
# write chosen patients/controls to a txt file
def save_selection_info(GENERAL_DIR, syn, trial):    
    ID_dir = GENERAL_DIR + "\\{}\\{}-selected-ID-controls".format(syn, syn)
    ID_files = [f for f in listdir(join(ID_dir)) if isfile(join(ID_dir, f)) and ".jpg" in f ]
   
    syn_dir = GENERAL_DIR + "\\{}\\{}-patients".format(syn, syn)
    syn_files = [f for f in listdir(join(syn_dir)) if isfile(join(syn_dir, f)) and ".jpg" in f ]
   
    selection_info = open("results/{}/{}-selection-info-run-{}.txt".format(syn, syn, trial), "w")   
                         
    selection_info.write("Patients for syndrome {}\n".format(syn))                        
    for syn in syn_files:
        selection_info.write(syn + "\n")
   
    selection_info.write("\nControls for syndrome {}\n".format(syn))
    for ID in ID_files:
        selection_info.write(ID + "\n")
                         
    selection_info.close()


def main(GENERAL_DIR, syn_list, trial):
    
    print("Selecting controls for trial {} \nfor syndroms: {}\n".format(trial, syn_list))
     
    for syn in syn_list:
                         
        # open relevant excel files
        df_syn = open_syn_excel(GENERAL_DIR, syn)
        df_ID = open_control_excel(GENERAL_DIR, syn)
        #make_hist(df_syn)
        
        # select controls 
        df_select_syn, df_select_ID = select_controls(GENERAL_DIR, syn, df_syn, df_ID)
        print("Syndrome {} \nShape of patient data {}, shape of found control data {}".format(syn, df_select_syn.shape, df_select_ID.shape))
        
        # write selected patients/controls to excel files 
        save_info_to_excel(GENERAL_DIR, syn, df_select_syn, df_select_ID)
        
        # save images from excel file to correct directories
        save_img_from_excel_controls(GENERAL_DIR, syn)
        save_img_from_excel_patients(GENERAL_DIR, syn)
        
        # save info about which patient/controls were selected in this run
        save_selection_info(GENERAL_DIR, syn, trial)    
                
    print("Done running ID_control_selection.py")

