import json
import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """Carrega o modelo do disco"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    """
    Transforma o JSON de entrada do Crew.ai em DataFrame para o modelo.
    Esperamos um JSON assim: 
    {"Age": 25, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.5, "BodyTemp": 98, "HeartRate": 70}
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        
        feature_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        
        df = pd.DataFrame(data)
        df = df[feature_columns]
        return df
    else:
        raise ValueError(f"Content type {request_content_type} not supported")

def predict_fn(input_data, model):
    """Faz a predição"""
    proba = model.predict_proba(input_data) 
    return proba

def output_fn(proba, response_content_type):
    """Formata a saída"""
    risco_prob = proba[0][1]

    res = int(proba[0][1] > 0.5)
    
    response = {
        "maternal_health_risk": bool(res == 1),
        "risk_probability": f"{risco_prob:.2%}"
    }
    
    return json.dumps(response)