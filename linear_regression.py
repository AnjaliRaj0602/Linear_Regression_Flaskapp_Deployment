import numpy as np
import matplotlib.pyplot
import pandas as pd
import pickle

dataset = pd.read_csv('50_Startups.csv')


def state_code_change(state):
    state_dict={
            'New York':0,
            'California':1,
            'Florida':2
            }
    return state_dict[state]

dataset['State']=dataset['State'].apply(lambda x: state_code_change(x))
    


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


pickle.dump(regressor,open('model.pkl','wb'))




model = pickle.load(open('model.pkl','rb'))


def predict(data):
    data[3]=state_code_change(data[3])
    return model.predict(np.reshape(np.array(data),(1,4)))

features=[160000,176000,165000,'California']
prediction=predict(features)

print(prediction)
    



