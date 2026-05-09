from flask import Flask, request, jsonify
import numpy as np
import torch
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from model_loader import model, scaler, JANELA

app = Flask(__name__)

# métricas Prometheus (passo 10)
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total de previsões')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latência da previsão')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    inicio = time.time()
    dados = request.get_json()
    precos = dados.get('precos')

    JANELA_PRECOS = JANELA + 1   # 61 preços geram 60 deltas

    if not precos or len(precos) != JANELA_PRECOS:
        return jsonify({
            'erro': f'Envie uma lista com {JANELA_PRECOS} preços (61 dias).'
        }), 400

    # converte preços em deltas
    arr_precos = np.array(precos).reshape(-1, 1)
    deltas = np.diff(arr_precos, axis=0)   # shape (60, 1)

    # normaliza os deltas
    deltas_norm = scaler.transform(deltas)

    # tensor pro modelo
    x = torch.tensor(deltas_norm, dtype=torch.float32).unsqueeze(0)  # (1, 60, 1)

    with torch.no_grad():
        pred_delta_norm = model(x).numpy()

    # desnormaliza o delta previsto
    pred_delta_real = scaler.inverse_transform(pred_delta_norm)[0][0]

    # reconstrói o preço final: preço de hoje + delta previsto
    preco_ontem = precos[-1]
    preco_previsto = preco_ontem + float(pred_delta_real)

    PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY.observe(time.time() - inicio)

    return jsonify({
        'preco_previsto': preco_previsto,
        'delta_previsto': float(pred_delta_real)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
