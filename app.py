from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

app = Flask(__name__)

#model = pd.read_pickle("model_linier.pickle")
model_file = open('modelnb.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    x1=float(request.form['x1'])
    x2=float(request.form['x2'])
    x3=float(request.form['x3'])
    x4=float(request.form['x4'])
    x5=float(request.form['x5'])
    x6=float(request.form['x6'])

    x=np.array([[x1,x2,x3,x4,x5,x6]])

    prediction = model.predict(x)
    if(prediction==0):
        output="Normal"
    else:
        output="Sakit"
    # output = round(prediction[0], 2)

    return render_template('index.html', hasil=output,x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)

# @app.route('/evaluate')
# def evaluate():
#     # Jika Anda memiliki data uji
#     X_test = model_file.drop('target_column', axis=1)
#     y_test = model_file['target_column']
#     predictions = model.predict(X_test)

#     # Akurasi prediksi
#     accuracy = accuracy_score(y_test, predictions)
#     return render_template('index.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)