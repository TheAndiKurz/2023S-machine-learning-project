# Insert the file into the path ./model.pkl

# Scored on JupyterHub:
# Train Dataset Score: 0.8107224556002434
# Test Dataset Score: 0.7905437463392786

import joblib

def leader_board_predict_fn(values):

    # Load the trained model from file
    model = joblib.load("./model.pkl")
    
    return model.predict(values)
