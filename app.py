import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if output == 1:
      return render_template('home.html', prediction_text='Patient should be Show Up ,\nShow Up : {}'.format(output))
    else:
      return render_template('home.html', prediction_text='Patient should not be Show Up ,\nShow Up : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
