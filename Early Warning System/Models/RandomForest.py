import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import numpy as np

def random_forest_result():
    data = pd.read_csv('./data.csv')
    data = data.sample(frac=0.9)


    X = data.drop(['SUBJECT_ID','TARGET'], axis=1)
    y = data['TARGET']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)

    # def objective_wrapper(feature_values):
    #     return objective_function(feature_values, rf)
    
    # # Your existing code for initial guess and bounds...
    # initial_guess = np.mean(X_train, axis=0)
    # bounds = [(X_train[col].min(), X_train[col].max()) for col in X_train.columns]
    
    # # Use the wrapper function in minimize
    # result = minimize(objective_wrapper, initial_guess, bounds=bounds, method='L-BFGS-B')
    # optimal_feature_values = result.x
    
    #print("Optimal feature values for class 1 prediction:", optimal_feature_values)
    

    return rf, X_train, y_test, y_pred, {"Accuracy" : accuracy, "Precision": precision, "Recall" : recall}

def objective_function(feature_values,model):
    # Reshape feature_values to match the input shape expected by the model
    feature_values_reshaped = feature_values.reshape(1, -1)
    # Predict_proba returns a list of [prob_class_0, prob_class_1]
    predicted_prob = model.predict_proba(feature_values_reshaped)[0][1]
    # Since we want to maximize prob_class_1, minimize its negative
    return -predicted_prob


