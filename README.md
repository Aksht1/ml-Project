# Crop Recommendation Demo (Frontend + Backend)

This is a minimal demo that adds a simple Flask backend and a static frontend to the project.

Run locally (PowerShell):

```powershell
python -m pip install -r requirements.txt
python webapp.py
```

Then open `http://localhost:5000` in your browser.

The `/api/predict` endpoint accepts JSON with fields `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall` and returns a mock recommendation. Replace the heuristic in `webapp.py` with your trained model when ready.
