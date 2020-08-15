# Face Classification 
Fien Ockers, march 2020

Project for my AI thesis regarding binary syndrome classification of faces of people with Intellectual Disability. 

The complete pipeline script follows the next steps:

1. Selecting ID controls and saving them as well as the patient images.
    run ID_control_selection.py
        - syndrome image data should be in: GENERAL_DIR\syn\syn-all-photos and is save to GENERAL_DIR\syn\syn-patients
        - control image data should be in: GENERAL_DIR\ID-controls
        - syndrome excel info should be in GENERAL_DIR\syn\syn_Database.csv
        - control excel info should be in GENERAL_DIR\ID-controls\all_ID_controls_info_complete.csv"
    
    
2. Saving the representations for all images in csv files.
    run save-representations-general.ipynb (check GENERAL_DIR)
    
3. Running the main classifier file which tries multiple classifiers (knn, svm, random forest, gradient boost and ada boost). 
    run classifiers-general.ipynb
