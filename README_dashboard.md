# STEA dashboard

Browser-based dashboard for exploring `STEA_2026.xlsx` (and similar exports).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run stea_dashboard.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## What it supports

- Totals per year: `Haettu` (applied), `Ehdotettu` (proposed), `Myönnetty` (approved)
- Filters: year range, keyword search (ALL/ANY), organization, avustuslaji type, kokonaisuus, järjestöluokka, geography substring
- Per-entry plots: applied vs approved per year using selectable “unique entry definition”
- Export: download filtered rows as CSV

## UI language

Use the sidebar toggle `Kieli / Language` to switch between Finnish and English UI labels.

## inotify / watchdog limits (Linux)

This repo includes `.streamlit/config.toml` to avoid `inotify instance limit reached` by using polling instead of watchdog.
If you still hit limits, you can temporarily disable watching entirely with:

```bash
streamlit run stea_dashboard.py --server.fileWatcherType none
```
