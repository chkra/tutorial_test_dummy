from datetime import datetime
from flask import Flask, request, render_template
from . import app

import numpy as np
import pickle

model = pickle.load(open('iris_model.pkl', 'rb')) # loading the trained model


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")


@app.route("/iris")
def iris():
    return render_template("predict.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # making prediction
    return render_template('predict.html', prediction_text='Predicted Class: {}'.format(prediction)) # rendering the predicted result