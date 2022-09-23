from fastapi import FastAPI, Request
from src.weight_models.model_loader import load_LSTM_model, LSTM_predict, load_GPT2_model, GPT2_predict

LSTM_model = load_LSTM_model()
print("INFO :: LSTM loaded ...")
GPT2_model = load_GPT2_model()
print("INFO :: GPT2 loaded ...")

app = FastAPI()

@app.post("/prediction/{model_type}")
async def nom_de_fonction_POST(body : Request, model_type : str):
    data = await body.json()
    input_data = data['phrase']
    if model_type == "LSTM":
        result = LSTM_predict(input_data, LSTM_model)
    elif model_type == "GPT2":
        result = GPT2_predict(input_data, GPT2_model)
    else:
        result = dict(message="Mauvais Modèle")
    return result
