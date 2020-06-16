#!/usr/bin/env python
# coding: utf-8

## pip install cmake; pip install dlib
import numpy as np
import argparse
import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile
import csv
import itertools
import dlib

def rect_to_bb(rect):
    
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def extract_features(keypoints): ## all possible combinations instead of 11 
    assert keypoints.shape == (68,2)
    feats = []
    denom = np.linalg.norm(keypoints[0]-keypoints[16])
    
    combs = [comb for comb in itertools.combinations([*range(0, len(keypoints)-1)], 2)]
    
    best_features = []
    #indices = [152, 164, 358, 1168, 1184, 1186, 1188, 1279, 1281, 1346, 1359, 1400, 1808, 2116]
    #print("AMount of combinations: {}".format(len(combs)))
    for comb in combs:
        a = comb[0]
        b = comb[1]
        
        #if not (a ==0 and b == 16):
        feats.append(np.linalg.norm(keypoints[a]-keypoints[b])/denom)
        
    return [], feats


def extract_features(image, keypoints, text): ## all possible combinations instead of 11 
    assert keypoints.shape == (68,2)
    feats = []
    denom = np.linalg.norm(keypoints[0]-keypoints[16])
    
    combs = [comb for comb in itertools.combinations([*range(0, len(keypoints))], 2)]
    for comb in combs:
        a = comb[0]
        b = comb[1]
        
        if not (a ==0 and b == 16):
            feats.append(np.linalg.norm(keypoints[a]-keypoints[b])/denom)
    
    px_size = 20
    text_feats = texture_feature(image, keypoints, px_size)    
    text_feats = [x / 255 for x in text_feats]
    if text:
        return [], feats+text_feats
    else:
        return [], feats

def texture_feature(image, keypoints, px_size):
    
    indices = [36, 39, 42, 45, 31, 33, 35, 48, 54]
    features = []

    for i in indices:
        keypoint = keypoints[i]
        (x, y) = keypoint
        small_image = image[y-px_size:y+px_size+1, x-px_size:x+px_size+1]

        median = cv2.medianBlur(small_image, 5)
        resized = cv2.resize(median, (10,10), interpolation = cv2.INTER_AREA)

        features.append(resized)

    return np.array(features).flatten().tolist()



def get_features(path, text):
    # load the input image, resize it, and convert it to grayscale

    image = cv2.imread(path)
    image = imutils.resize(image, width=500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(image, 1)
    for (i, rect) in enumerate(rects):
        
        shape = predictor(image, rect)
        shape = shape_to_np(shape)

        keypoints, feats = extract_features(image, shape, text)
    
        return feats
    print("No face found")
    print(path)
    return np.zeros(11).tolist()

detector = dlib.get_frontal_face_detector()
path_to_shape_predictor = r"C:/Users/manz616236/Documents/face-classification/binary-classification/models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path_to_shape_predictor)

def dlib_landmarks_reps(GENERAL_DIR, syn_name):

    # if text == True, then dlib-text features are generated and otherwise just dlib
    text = False
    syn_rep, ID_rep = [], []

    syn_dir = GENERAL_DIR + "\\{}\{}-patients".format(syn_name, syn_name)
    ID_dir = GENERAL_DIR + "\\{}\{}-selected-ID-controls".format(syn_name, syn_name)
    
    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn_name in f] 
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".jpg" in f]
    
    print("Syn_list: {}, ID_list: {}".format(len(files_syn), len(files_ID)))

        
    # for each kdv image save deepface rep as list:
    for filename in files_syn:
        feats = get_features(join(syn_dir, filename), text)
        syn_rep.append([filename] + feats) 


    # for each ID image save deepface rep as list:
    for filename in files_ID:
        feats = get_features(join(ID_dir, filename), text)
        ID_rep.append([filename] + feats)  
        
        
    print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))
 
    if text:
        method = "dlib-text"

    else:
        method = "dlib"

    # location to save representation
    csv_file_syn = GENERAL_DIR + "\\{}\\representations\\{}-patients-{}.csv".format(syn_name, syn_name, method)
    csv_file_ID = GENERAL_DIR + "\\{}\\representations\\ID-controls-{}.csv".format(syn_name, method)


    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)

    print("Done with saving all {} representations for {}.".format(method, syn_name))




