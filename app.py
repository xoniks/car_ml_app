import streamlit as st 
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

def load_model(file_name):
    load_model = pickle.load(open(file_name, 'rb'))
    return load_model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def main():
    
    st.title('ML car app')
    model_file = 'egezon_car_ml.pkl'
    model = load_model(model_file)
    
    st.header('Enter values for prediction:')
    x_value = st.text_input('X values')
    x_value = x_value.split(',')
    print(x_value)
    x_value = [int(i) for i in x_value]
    print(x_value)
    submit_button = st.button('Predict')
    
    if submit_button:
        
        X_test = [x_value]
        y_predict = predict(model,X_test)
        st.write(f'The predicted value for {X_test} is {y_predict} ')
        
if __name__ == '__main__':
    main()