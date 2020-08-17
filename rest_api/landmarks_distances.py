from os.path import join, isfile
from os import listdir
import numpy as np
import itertools
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append("..")
from global_variables import LEFT, RIGHT

def get_distances(keypoints):
    feats  = []
    combs = [comb for comb in itertools.combinations([*range(0, len(keypoints))], 2)]
    for comb in combs:
        a = comb[0]
        b = comb[1]
        feats.append(distance.euclidean(keypoints[a], keypoints[b]))
    return feats


def get_features(rep):
    landmarks_left, landmarks_right = [], []
    for count, landmark in enumerate(rep):
        if count in LEFT:
            landmarks_left.append((float(landmark[0]), float(landmark[1]), float(landmark[2])))
        if count in RIGHT:
            landmarks_right.append((float(landmark[0]), float(landmark[1]), float(landmark[2]))) 
            
    feats_left = get_distances(landmarks_left)
    feats_right = get_distances(landmarks_right)

    all_feats = feats_left + feats_right
    return all_feats