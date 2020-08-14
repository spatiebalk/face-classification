from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

import pickle5 as pickle
import os
from os.path import isfile, join
import numpy as np
import tensorflow as tf 
from statistics import mode

import deepface
import facereader
import pointnet
import landmarks_distances

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
syns = ['ADNP', 'ANKRD11','CDK13', 'DEAF1', 'DYRK1A', 'EHMT1', 'FBXO11', 'KDVS', 'SON', 'WAC', 'YY1', '22q11']
deepface_model = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    global deepface_model
    if deepface_model == None:
       print("creating deepface")
       deepface_model = deepface.create_deepface()
       
    if request.method == 'POST':
        syn = request.form.get('syns')
        
        if 'file' not in request.files:
            print('No file part')

        else:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("selected syn", syn)
        return analyze(syn, file.filename)
    
    return render_template('home.html', syns=syns)


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/contact")
def contact():
    return render_template('contact.html', title='Contact')


@app.route("/analyze")
def analyze(syn, filename):
    deepface_rep = deepface.get_deepface_rep(deepface_model, join(UPLOAD_FOLDER, filename))
    facereader_rep = facereader.get_facereader_rep(join(UPLOAD_FOLDER, filename))
    
    prob_df, pred_df, prob_pn, pred_pn, prob_rf, pred_rf, mean_prob, mode_pred = 0, 0, 0, 0, 0, 0, 0, 0 
    
    if isinstance(facereader_rep, np.ndarray):
        facereader_rep_found=True
    else:
        facereader_rep_found=False
        prob_pn = facereader_rep
        pred_pn = facereader_rep
        prob_rf = facereader_rep
        pred_rf = facereader_rep
        
        
    #get deepface prob
    model_df = pickle.load(open('models/knn-deepface-{}'.format(syn), 'rb'))
    prob_df = model_df.predict_proba(deepface_rep[0].reshape(1, -1))[:,1][0]
    pred_df = model_df.predict(deepface_rep[0].reshape(1, -1))[0]
    
    if facereader_rep_found:
        # get pointnet prob
        model_pn = pointnet.generate_model()
        model_pn.load_weights('models/pointnet-{}.ckpt'.format(syn))
        facereader_rep_pn = np.expand_dims(facereader_rep, axis=0)
        y_pred_array = model_pn.predict(facereader_rep_pn)
        prob_pn = y_pred_array[0][1]
        pred_pn = tf.math.argmax(y_pred_array, -1).numpy()[0]
        
        # get random forest prob        
        model_rf = pickle.load(open('models/randomforest-{}'.format(syn), 'rb'))
        distances = landmarks_distances.get_features(facereader_rep)
        prob_rf = model_rf.predict_proba(np.array(distances).reshape(1, -1))[:,1][0]
        pred_rf = model_rf.predict(np.array(distances).reshape(1, -1))[0]

        # get mean prob and mode pred
        mean_prob = np.mean([prob_df, prob_pn, prob_rf])
        mode_pred = mode([pred_df, pred_pn, pred_rf])
        

    else:
        mean_prob = prob_df
        mode_pred = pred_df
        
    ## visualize scores 
    return render_template('analyze.html', title='Analyze', syn=syn, filename=filename, prob_df=prob_df, pred_df=pred_df, prob_pn=prob_pn, pred_pn=pred_pn, prob_rf=prob_rf, pred_rf=pred_rf, mean_prob=mean_prob, mode_pred=mode_pred)

if __name__ == '__main__':
    app.run(debug=True)