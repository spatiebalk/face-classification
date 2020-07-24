#!/usr/bin/env python
# coding: utf-8

import openface_cfps
import deepface
import facereader

def main(GENERAL_DIR, syn_list):

    for syn in syn_list:
        print(syn, "\n")
        
        print ("openface and cfps")
        openface_cfps.openface_cfps_reps(GENERAL_DIR, syn)

        print ("deepface")
        deepface.deepface_reps(GENERAL_DIR, syn)
    
        print ("facereader-landmarks")
        facereader.facereader_landmarks_reps(GENERAL_DIR, syn) 

    print("Done running save_representations.py")

