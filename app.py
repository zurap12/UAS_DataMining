
import imp
from pyexpat import features

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
# label_encoder = preprocessing.LabelEncoder()
# df = pd.read_csv(".\data\Churn_Modelling.csv")
# categorical_df = df[[ "Surname","Geography","Gender"]]


app = Flask(__name__)
model = pickle.load(open("modelGnb.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # for x in request.form.values():
    #     data = []
    #     if (type(x) == str):

    #     data.append(x)
    float_features = [float(x) for x in request.form.values()]
    # for label in categorical_df[0:]:
    #     df[label]= label_encoder.fit_transform(df[label])
    #     df = df.astype({label : 'float'})
    from sklearn.preprocessing import minmax_scale
    features = [minmax_scale(np.array(float_features))]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)