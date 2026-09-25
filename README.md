# Intelligent AML Monitoring & SAR Assistant

Transaction monitoring for anti-money-laundering (AML) that combines:

- **Risk classifier**: XGBoost that scores each transaction's laundering probability.
- **Anomaly detector**: Isolation Forest, with an optional autoencoder, that flags statistical outliers.
- **Explainability**: SHAP and LIME contributions for the risk score.
- **SAR generation**: an LLM (DeepSeek) drafts a Suspicious Activity Report narrative from the three model outputs above.
- **KYC / EDD review**: an LLM checks customer due-diligence documents.
- **RAG assistant**: question answering over an AML regulations corpus.
- **FastAPI service** and **Streamlit dashboard**.

## Setup

```bash
conda create -n AML python=3.11
conda activate AML
pip install -r requirements.txt
```

API keys go in a git-ignored `.env` file at the project root (copy `.env.example`). Real environment variables work too. Keys are never stored in `config.yaml`.

| Variable | Used by |
|---|---|
| `DEEPSEEK_API_KEY` | SAR generation, KYC validation, EDD summaries, RAG answers |
| `OPENAI_API_KEY` | RAG embeddings |

Paths, risk-level thresholds and the LLM endpoint are set in [config.yaml](config.yaml).

## Data

This project uses the [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) dataset, which is not tracked in git. Place the files as follows:

- `data/raw/archive/LI-Small_Trans.csv`: raw file, used to train the risk classifier.
- `data/processed/LI-Small_Trans.csv`: cleaned copy with underscore column names, produced by `notebooks/01_data_exploration.ipynb` and used by the anomaly detector.
- `data/kyc_docs/*.txt`: KYC documents (`Field: value` lines).
- `data/regulations_corpus/*.txt|*.pdf`: regulations and guidance for the RAG assistant. This folder is currently empty.

## Training

Run these from the project root:

```bash
python src/models/risk_classifier.py     # ~5 min on LI-Small (6.9M rows)
python src/models/anomaly_detector.py    # Isolation Forest + autoencoder
```

The risk classifier:
- splits the data 80/20 into train and test, then holds out 15% of train for validation;
- uses the validation set for early stopping and for the decision threshold (optimised for F2);
- reports metrics only on the untouched test set.

Account-history features (velocity, running mean/std, distinct counterparties) use only *earlier* transactions. They are therefore computed the same way in training and when scoring a single live transaction.

Latest held-out results on LI-Small (see [risk_classifier_output.txt](risk_classifier_output.txt)):

| Metric | Value |
|---|---|
| ROC AUC | 0.971 |
| PR AUC (positive rate 0.05%) | 0.067 |
| Precision / Recall at tuned threshold | 0.076 / 0.32 |

## Running

```bash
python app/api.py                  # FastAPI on http://127.0.0.1:8000 (docs at /docs)
streamlit run app/dashboard.py     # dashboard; talks to the API
```

Main endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /predict-risk`, `/batch-risk-predict` | Model risk score, level, `is_suspicious` and `scoring_method` (`model`, or `rules` if no trained model is available) |
| `POST /analyze-comprehensive` | Risk score, anomaly result and SHAP explanation |
| `POST /generate-sar`, `/generate-sar-full` | SAR narrative built from that analysis |
| `POST /validate-kyc` | Upload a KYC `.txt`. Fields are extracted, then reviewed by the LLM |
| `GET /model-info` | Which models are loaded |

Risk levels (LOW / MEDIUM / HIGH) come from `thresholds` in `config.yaml`. `is_suspicious` uses the classifier's tuned threshold.

## Tests

```bash
pytest
```

The tests use small synthetic datasets and a stubbed LLM, so they need neither the IBM data nor API keys.

## Layout

```
app/                 FastAPI service and Streamlit dashboard
src/data/            data loading, KYC parsing
src/models/          risk classifier, anomaly detector, SHAP/LIME explainer
src/llm/             SAR generator, KYC validator, EDD processor, RAG assistant
src/realtime/        replay a transactions CSV as a stream and alert on suspicious rows
src/utils/           config loading
models/              trained model artifacts
notebooks/           exploration and experiments
tests/               pytest suite
```

## Known limitations

- The API scores each transaction on its own, without looking up the account's history, so every live transaction looks like the account's first. A feature store of recent per-account activity would let the velocity and history features take effect at serving time.
- The models only know numeric bank IDs from the dataset. Bank names sent by the dashboard are treated as unknown banks.
- Jurisdiction, KYC completeness and beneficiary type are not model inputs. The SAR generator reports them as rule-based red flags alongside the model score.
- The RAG assistant needs documents added to `data/regulations_corpus/`.
