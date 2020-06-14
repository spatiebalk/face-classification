#!/usr/bin/env python
# coding: utf-8

import pandas
import csv 
from os.path import join, isfile
from os import listdir, path
import pandas as pd
import xlrd
import numpy as np
import keras
from PIL import Image 
from keras.models import Model
import tensorflow
from tqdm import tqdm
import dlib

import cfps_openface
import deepface
import dlib_landmarks
import facereader


def main(GENERAL_DIR, syn_list):

    for syn in tqdm(syn_list):

        # openface
        cfps_openface.openface_cfps_reps(GENERAL_DIR, "openface", syn)

        # cfps
        cfps_openface.openface_cfps_reps(GENERAL_DIR, "cfps", syn)

        # deepface
        deepface.deepface_reps(GENERAL_DIR, syn)

        # dlib
        dlib_landmarks.dlib_landmarks_reps(GENERAL_DIR, syn)

        # facereader
        facereader.facereader_reps(GENERAL_DIR, syn) 

    print("Done running save_representations.py")
