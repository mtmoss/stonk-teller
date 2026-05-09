from flask import Flask, request, jsonify, render_template_string
import numpy as np
import torch
import time
import logging
from collections import deque
from model_loader import model, scaler, JANELA

# ============ logging ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============ métricas em memória ============
predictions_count = 0
latencies = deque(maxlen=1000)   # últimas 1000 latências
errors_count = 0
start_time = time.time()


# ============ rotas ============

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    global predictions_count, errors_count

    inicio = time.time()
    dados = request.get_json()
    precos = dados.get('precos') if dados else None

    JANELA_PRECOS = JANELA + 1

    if not precos or len(precos) != JANELA_PRECOS:
        errors_count += 1
        logger.warning('requisição inválida: %s preços recebidos', len(precos) if precos else 0)
        return jsonify({
            'erro': f'Envie uma lista com {JANELA_PRECOS} preços (61 dias).'
        }), 400

    try:
        arr_precos = np.array(precos).reshape(-1, 1)
        deltas = np.diff(arr_precos, axis=0)
        deltas_norm = scaler.transform(deltas)
        x = torch.tensor(deltas_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_delta_norm = model(x).numpy()

        pred_delta_real = scaler.inverse_transform(pred_delta_norm)[0][0]
        preco_ontem = precos[-1]
        preco_previsto = preco_ontem + float(pred_delta_real)

        latencia = time.time() - inicio
        predictions_count += 1
        latencies.append(latencia)

        logger.info('previsão #%d ok | latência %.3fs | delta %.4f',
                    predictions_count, latencia, pred_delta_real)

        return jsonify({
            'preco_previsto': round(preco_previsto, 4),
            'delta_previsto': round(float(pred_delta_real), 4),
            'latencia_segundos': round(latencia, 4)
        })
    except Exception as e:
        errors_count += 1
        logger.error('erro na previsão: %s', e)
        return jsonify({'erro': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({
        'predictions_total': predictions_count,
        'errors_total': errors_count,
        'latency_avg_seconds': round(sum(latencies) / len(latencies), 4) if latencies else 0,
        'latency_max_seconds': round(max(latencies), 4) if latencies else 0,
        'uptime_seconds': round(time.time() - start_time, 1)
    })


# ============ página HTML simples ============

PAGINA_HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>Stonk Teller 📈🔮</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
           max-width: 720px; margin: 40px auto; padding: 20px;
           color: #2D3D6A; background: #FFFFFF; }
    h1 { color: #2D3D6A; }
    textarea { width: 100%; height: 140px; font-family: monospace;
               border: 2px solid #2D3D6A; border-radius: 6px; padding: 10px; }
    button { background: #FF8673; color: white; border: none;
             padding: 12px 24px; font-size: 16px; border-radius: 6px;
             cursor: pointer; margin-top: 10px; }
    button:hover { background: #2D3D6A; }
    .resultado { background: #FACD73; padding: 16px; border-radius: 6px;
                 margin-top: 20px; display: none; }
    .resultado.erro { background: #ffdddd; }
    code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>Stonk Teller 📈🔮</h1>
  <p>LSTM que prevê o preço de fechamento da AAPL com base nos últimos 61 dias.</p>

  <h3>1. Cole 61 preços (separados por vírgula)</h3>
  <p style="font-size: 13px;">
    Tip: pega os 61 últimos preços de fechamento de uma ação.
  </p>
  <textarea id="precos" placeholder="180.5, 181.2, 179.8, ..."></textarea>

  <button onclick="prever()">Prever próximo preço</button>
  <button onclick="exemplo()" style="background: #2D3D6A;">Carregar exemplo</button>

  <div id="resultado" class="resultado"></div>

  <hr style="margin-top: 40px;">
  <p style="font-size: 13px;">
    Endpoints: <code>POST /predict</code> · <code>GET /metrics</code> · <code>GET /health</code><br>
    Repo: <a href="https://github.com/mtmoss/stonk-teller">github.com/mtmoss/stonk-teller</a>
  </p>

  <script>
    async function prever() {
      const texto = document.getElementById('precos').value;
      const precos = texto.split(/[,\\s]+/).filter(x => x).map(Number);
      const div = document.getElementById('resultado');
      div.style.display = 'block';

      if (precos.length !== 61 || precos.some(isNaN)) {
        div.className = 'resultado erro';
        div.innerHTML = '<b>Erro:</b> precisa de exatamente 61 preços numéricos. Você enviou ' + precos.length + '.';
        return;
      }

      div.className = 'resultado';
      div.innerHTML = 'Calculando...';

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({precos})
        });
        const data = await res.json();
        if (data.erro) {
          div.className = 'resultado erro';
          div.innerHTML = '<b>Erro:</b> ' + data.erro;
        } else {
          div.innerHTML =
            '<b>Preço previsto:</b> $' + data.preco_previsto + '<br>' +
            '<b>Variação prevista:</b> ' + (data.delta_previsto >= 0 ? '+' : '') + '$' + data.delta_previsto + '<br>' +
            '<b>Latência:</b> ' + data.latencia_segundos + 's';
        }
      } catch (e) {
        div.className = 'resultado erro';
        div.innerHTML = '<b>Erro de rede:</b> ' + e.message;
      }
    }

    function exemplo() {
      const fake = Array.from({length: 61}, (_, i) => (180 + i * 0.3).toFixed(2));
      document.getElementById('precos').value = fake.join(', ');
    }
  </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(PAGINA_HTML)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))   # Render passa PORT via env
    app.run(host='0.0.0.0', port=port)