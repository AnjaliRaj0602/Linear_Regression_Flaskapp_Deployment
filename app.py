import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

def state_code_change(state):
    state_dict={
            'New York':0,
            'California':1,
            'Florida':2
            }
    return state_dict[state]

def predicts(data):
    data[3]=state_code_change(data[3])
    return model.predict(np.reshape(np.array(data),(1,4)))


@app.route('/')
def home():
    return render_template('regressionindex1.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    features=[x for x in request.form.values()] 
 
    for i in range(len(features)-1):
        features[i]=int(features[i])
     
    prediction= predicts(features)
    
    
    
    return render_template('regressionindex1.html', prediction_text =  'Employee salary should be $ {}'.format(prediction))


if __name__=="__main__":
    app.run(debug=True)