import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path

# raiz do projeto = pasta acima de /app
BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = BASE_DIR / 'saved_model'

class ModeloLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.linear(out)

with open(SAVED_MODEL_DIR / 'metadados.json') as f:
    META = json.load(f)

model = ModeloLSTM(hidden_size=META['hidden_size'], num_layers=META['num_layers'])
model.load_state_dict(torch.load(SAVED_MODEL_DIR / 'model.pth'))
model.eval()

scaler = joblib.load(SAVED_MODEL_DIR / 'scaler.pkl')
JANELA = META['janela']