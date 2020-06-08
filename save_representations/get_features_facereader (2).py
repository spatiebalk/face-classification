# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:05:20 2020

@author: manz184215
"""
def get_data_facereader(file):
    from System import Drawing
    import json
    FR.GrabCredits(1)
    bitmap = Drawing.Bitmap(file)
    try:
        result = (FR.AnalyzeFace(bitmap))
        result = json.loads(result.ToJson())
    except:
        print("An exception occurred")
        result = 'error'
    return result

import clr
clr.AddReference("\\\\umcfs020\\antrgdata$\\Genetica Projecten\\Facial Recognition\\Studenten en Onderzoekers\\Lex\\Projecten\\FaceReader\\Python\\FaceReaderSDK\\FaceReaderSDK.dll")
from VicarVision import FaceReader
clr.AddReference('System.Drawing')
FR = FaceReader.FaceReaderSDK("0CF99D")
   
results_kdvs = []

import glob
for file in glob.glob('H:\Genetica Projecten\Facial Recognition\Syndromen\KdVs\photos dataset Roos\*.jpg'):
    print(file)
    results_kdvs.append(get_data_facereader(file))

        
