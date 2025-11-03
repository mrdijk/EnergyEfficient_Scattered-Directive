import io
import pandas as pd
import joblib
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
data_store = []  # list of dataframes
model = None

class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    data_store.append(df)
    return {"message": f"Received data with shape {df.shape}"}

@app.post("/train_model/")
def train_model():
    global model

    if not data_store:
        return {"error": "No data"}

    df = pd.concat(data_store, ignore_index=True)
    if "Survived" not in df.columns:
        return {"error": "Data does not contain label"}
    
    X = df.drop("Survived", axis=1)
    y = df['Survived']

    model = ServerModel(X.shape[1])