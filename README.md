Dash AI Reversal Detector

Quick start:

1. Create a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate

2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   python dash_app.py

Open http://localhost:8050 in your browser.

Notes:
- The app compares NASDAQ (^IXIC) vs DJIA (^DJI) and flags percentage-change reversals in the NASDAQ/DJIA ratio.
- Adjust the period and threshold in the sidebar.
