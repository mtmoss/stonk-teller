# Stonk Teller 📈🔮

Modelo LSTM para prever o preço de fechamento da AAPL.
Tech Challenge Fase 4 — FIAP MLET.

## Arquitetura

- **Modelo:** LSTM com 1 camada (50 unidades hidden) + camada linear de saída
- **Alvo:** variação diária do preço (delta), não o preço bruto
- **Input da API:** lista de 61 preços (gera 60 deltas)
- **Output:** delta previsto + preço previsto reconstruído
- **Métricas no conjunto de validação:** MAE $2.75, RMSE $4.03, MAPE 1.18%

## Stack

- Python 3.11, PyTorch
- Flask (API REST)
- Docker + Docker Compose
- Prometheus + Grafana (monitoramento)

## Estrutura
stonk-teller/
├── data/                # CSV histórico da AAPL (Stooq)
├── notebooks/           # treinamento e troubleshooting documentado
├── saved_model/         # model.pth, scaler.pkl, metadados.json
├── app/                 # API Flask
├── monitoring/          # config Prometheus
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

## Como rodar

Pré-requisitos: Docker Desktop instalado.

```bash
git clone https://github.com/mtmoss/stonk-teller.git
cd stonk-teller
docker-compose up --build
```

Sobe três containers:
- API em `http://localhost:5001`
- Prometheus em `http://localhost:9090`
- Grafana em `http://localhost:3000` (login: admin / admin)

## Endpoints da API

### `GET /health`
```bash
curl http://localhost:5001/health
# → {"status": "ok"}
```

### `POST /predict`
Espera lista de **61 preços** (gera 60 deltas pra alimentar a LSTM).

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"precos": [180.0, 180.5, ..., 200.1]}'

# → {"delta_previsto": 0.18, "preco_previsto": 200.28}
```

### `GET /metrics`
Endpoint exposto pro Prometheus.
- `predictions_total`: contador de previsões feitas
- `prediction_latency_seconds`: histograma de latência

## Observações

A API espera 61 preços (e não 60) porque o modelo prevê **variação diária** (delta) em vez de preço bruto. Isso resolve o problema de extrapolação que afeta LSTMs treinadas em séries temporais com tendência. Detalhes do raciocínio em `notebooks/01_treinamento.ipynb`.

## Coleta dos dados

Dataset baixado manualmente do Stooq (`https://stooq.com/q/d/?s=aapl.us`). A biblioteca `yfinance` recomendada no enunciado estava com bugs de bloqueio do Yahoo durante o desenvolvimento (maio/2026).

## Vídeo de demonstração

[link do YouTube aqui — preencher depois]