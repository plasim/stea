# STEA Streamlit dashboard

This repo contains a Streamlit app (`stea_dashboard.py`) for exploring STEA funding data in Excel format. By default it loads `STEA_2026.xlsx` from the repo root, or you can upload another `.xlsx` in the sidebar.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run stea_dashboard.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (make sure `stea_dashboard.py`, `STEA_2026.xlsx`, `requirements.txt`, and `.streamlit/config.toml` are committed).
2. Go to https://share.streamlit.io → **New app**.
3. Pick your GitHub repo + branch, and set **Main file path** to `stea_dashboard.py`.
4. Deploy.

Notes:
- `requirements.txt` is what Streamlit Cloud uses to install Python dependencies.
- `runtime.txt` pins the Python version (optional, but recommended).
- If you don’t want to commit the Excel, remove `STEA_2026.xlsx` and rely on the sidebar uploader instead.
