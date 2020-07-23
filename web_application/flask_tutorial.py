from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

import pickle5 as pickle
import os
from os.path import isfile, join
import numpy as np
import keras

import deepface
import facereader

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        syndromes = request.form.getlist('syn')

        if 'file' not in request.files:
            print('No file part')

        else:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return analyze(syndromes, file.filename)

    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/contact")
def contact():
    return render_template('contact.html', title='Contact')


@app.route("/analyze")
def analyze(syns, filename):
    deepface_rep = deepface.get_deepface_rep(join(UPLOAD_FOLDER, filename))
    facereader_rep = facereader.get_facereader_rep(join(UPLOAD_FOLDER, filename))
    
    probs_df, probs_pn, probs_rf = [], [], [] 
    print(type(facereader_rep))
    
    if isinstance(facereader_rep, np.ndarray):
        facereader_rep_found=True
    else:
        facereader_rep_found=False
        probs_pn.append("error2")
        probs_rf.append("error2")
        

    for syn in syns:
        #get deepface prob
        model_df = pickle.load(open('models/knn-deepface-{}'.format(syn), 'rb'))
        result_df = model_df.predict_proba(deepface_rep[0].reshape(1, -1))[:,1]
        probs_df.append(result_df)
        
        if facereader_rep_found:
            # get poitnet prob
            print("here") 
            model_pn = keras.models.load_model('models/pointnet-{}'.format(syn))        
            y_pred_array = model_pn.predict(facereader_rep)
            probs_pn.append(y_pred_array[0][1])
        
            # get random forest prob        
            model_rf = pickle.load(open('models/randomforest-{}'.format(syn), 'rb'))
            result_rf = model_rf.predict_proba(facereader_rep)[:,1]
            probs_rf.append(result)

    ## visualize scores 
    indices = list(range(len(probs_df)))
    return render_template('analyze.html', title='Analyze', syns=syns, filename=filename, probs_df=probs_df, probs_pn=probs_pn, probs_rf=probs_rf, indices=indices)

if __name__ == '__main__':
    app.run(debug=True)