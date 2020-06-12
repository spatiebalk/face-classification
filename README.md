# Face Classification 
Fien Ockers, march 2020

Project for my AI thesis regarding binary syndrome classification of faces of people with Intellectual Disability. 

General steps:
1. Selecting ID controls and saving them as well as the patient images.
    run ID-control-selection.ipynb (check GENERAL_DIR and directory structure)
2. Saving the representations for all images in csv files.
    run save-representations-general.ipynb (check GENERAL_DIR)
3. Running the main classifier file which tries multiple classifiers (knn, svm, random forest, gradient boost and ada boost). 
    run classifiers-general.ipynb
