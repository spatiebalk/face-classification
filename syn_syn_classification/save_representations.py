import openface_cfps
import deepface
import facereader


def main(GENERAL_DIR, syn, control):
    
    print(syn, control, "\n")

    print ("openface and cfps")
    openface_cfps.openface_cfps_reps(GENERAL_DIR, syn, control)

    print ("deepface")
    deepface.deepface_reps(GENERAL_DIR, syn, control)

    print ("facereader-landmarks")
    facereader.facereader_landmarks_reps(GENERAL_DIR, syn, control) 

    print ("facereader-landmarks-distances")
    facereader.facereader_landmarks_dis_reps(GENERAL_DIR, syn, control) 

