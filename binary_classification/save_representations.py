#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm

import cfps_openface
import deepface
import facereader

def main(GENERAL_DIR, syn_list):

    for syn in tqdm(syn_list):
#         print ("\nopenface")
#         # openface
#         cfps_openface.openface_cfps_reps(GENERAL_DIR, "openface", syn)
#         print ("\ncfps")
#         # cfps
#         cfps_openface.openface_cfps_reps(GENERAL_DIR, "cfps", syn)
        
        print ("\ndeepface")
        # deepface
        deepface.deepface_reps(GENERAL_DIR, syn)
        #deepface_segmented.deepface_reps(GENERAL_DIR, syn)
        
#         print ("\nfacereader")
#         # facereader
#         facereader.facereader_reps(GENERAL_DIR, syn) 
#         print ("\nfacereader-landmarks")
#         # facereader
#         facereader.facereader_landmarks_reps(GENERAL_DIR, syn) 

    print("Done running save_representations.py")

