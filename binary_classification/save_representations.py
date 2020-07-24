#!/usr/bin/env python
# coding: utf-8

import cfps_openface
import deepface
import facereader

def main(GENERAL_DIR, syn_list):

    for syn in syn_list:
        print(syn)
        
        print ("\nopenface and cfps")
        # openface and cfps
        cfps_openface.openface_cfps_reps(GENERAL_DIR, syn)

#         print ("\ndeepface")
#         # deepface
#         deepface.deepface_reps(GENERAL_DIR, syn)
        
#         print ("\nfacereader")
#         # facereader
#         facereader.facereader_reps(GENERAL_DIR, syn) 
#         print ("\nfacereader-landmarks")
#         # facereader
#         facereader.facereader_landmarks_reps(GENERAL_DIR, syn) 

    print("Done running save_representations.py")

