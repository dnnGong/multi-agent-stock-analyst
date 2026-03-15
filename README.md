# app_test

This folder is a deployment-ready Streamlit stock-analysis app.

## 1) Install

```bash
cd app_test
python -m pip install -r requirements.txt
```

## 2) Configure env vars

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ALPHAVANTAGE_API_KEY="your_alphavantage_api_key"
# Optional:
# export STOCK_AGENTS_DB_PATH="/absolute/path/to/stocks.db"
```

## 3) Prepare database

Put your sqlite file at:
- `app_test/data/stocks.db`

If missing, the app will stop and show an error.

## 4) Run

```bash
streamlit run app.py
```
