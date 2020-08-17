import clr
import numpy as np
import json 
from collections.abc import Mapping
from os.path import join, isfile

import sys
sys.path.append("..")
from global_variables import SDK_ref, license

clr.AddReference(SDK_ref)
from VicarVision import FaceReader
clr.AddReference('System.Drawing')
from System import Drawing

FR = FaceReader.FaceReaderSDK(license)

def get_facereader_rep(filename):
    
    FR.GrabCredits(1)        
    bitmap = Drawing.Bitmap(filename)
    try:
        result = (FR.AnalyzeFace(bitmap))
        result = json.loads(result.ToJson())
    except:
        return "error" 
    
    if isinstance(result, Mapping):
        if result['FaceAnalyzed']:
            landmarks = []
            landmarks_dict = result['Landmarks3D']
            for item in landmarks_dict:
                landmarks.append([item['X'], item['Y'], item['Z']])
            landmarks = np.array(landmarks)
            return landmarks
        else:
            # no face found
            return "no face found"

