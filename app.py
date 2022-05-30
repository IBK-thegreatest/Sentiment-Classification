from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

vectorizer = pickle.load(open('transform.pkl', 'rb'))
model = pickle.load(open('NLP_Model.pkl','rb'))

app = Flask(__name__)

#Displaying the home route
@app.route('/')
def home():
    return render_template('home.html')

#Displaying the predict route for showing the predictions
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        vect = vectorizer.transform([message])
        my_prediction = model.predict(vect)

    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)