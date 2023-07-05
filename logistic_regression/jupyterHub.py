# Insert the file into the path ./model.pkl

import joblib

def leader_board_predict_fn(values):

    # Load the trained model from file
    model = joblib.load("./model.pkl")
    
    return model.predict(values)
