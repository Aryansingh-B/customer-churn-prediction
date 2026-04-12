import pickle 
import pandas as pd

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

def predict_churn(input_data: dict):
    """"
    input_data: dictionary of user inputs
    """
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return prediction, probability