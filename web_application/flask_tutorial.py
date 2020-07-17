from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

import pickle5 as pickle
import os
import deepface
from os.path import isfile, join
import numpy as np

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
def analyze(syndromes, filename):
    ## get score of deepface
    deepface_rep = deepface.get_deepface_rep(join(UPLOAD_FOLDER, filename))
    print(np.array(deepface_rep).shape)
    ## get score of knn classifier
    # load the model from disk
    probabilities = []
    for syn in syndromes:
    
        loaded_model = pickle.load(open('models/knn-deepface-{}'.format(syn), 'rb'))
        result = loaded_model.predict_proba(deepface_rep[0].reshape(1, -1))[:,1]
        probabilities.append(result)

    ## visualize scores 
    indices = list(range(len(probabilities)))
    return render_template('analyze.html', title='Analyze', syndromes=syndromes, filename=filename, probabilities=probabilities, indices=indices)

if __name__ == '__main__':
    app.run(debug=True)