# India Rainfall Dashboard (GeoRain Explorer)

Interactive Bokeh Server app for exploring monthly rainfall across Indian subdivisions.
It melts `SUBDIVISION + YEAR + JAN..DEC + Latitude/Longitude` into a time series,
maps points on a web map (lon/lat in °), and provides:
- Date-range and subdivision filtering
- Text search by station/subdivision
- Distribution histogram
- Top-N total rainfall view
- Tap on map → see monthly time series

## Repo Layout
```
.
├─ app/
│  ├─ main.py
│  ├─ theme.yaml
│  └─ data/
│     └─ Rainfall_Data_LL.csv
├─ requirements.txt
├─ Procfile
└─ render.yaml   # optional for Render
```

## Run Locally
```bash
pip install -r requirements.txt
bokeh serve app --port 5010   --allow-websocket-origin=localhost:5010   --allow-websocket-origin=127.0.0.1:5010   --show
```
Open http://localhost:5010/app

## Deploy on Render (free)
1) Push this repo to GitHub.
2) Render → New → Web Service → connect repo.
3) Build: `pip install -r requirements.txt`
4) Start: `bokeh serve app --address=0.0.0.0 --port=$PORT --use-xheaders --allow-websocket-origin=<HOST>`
5) After first deploy, replace `<HOST>` with your Render host (e.g., `india-rainfall-dashboard.onrender.com`) and redeploy.
Visit `https://<HOST>/app`

## Deploy on Railway (free credits)
Set Start Command:
```
bokeh serve app --address=0.0.0.0 --port=$PORT --use-xheaders --allow-websocket-origin=<YOUR-SUBDOMAIN>.up.railway.app
```
Visit `https://<YOUR-SUBDOMAIN>.up.railway.app/app`

### Notes
- The `--allow-websocket-origin` value must be the exact host (no `https://`).
- Because the app is served as `bokeh serve app`, your path is `/app`.
=======
# india-rainfall-dashboard
Interactive Bokeh Server app to explore subdivision rainfall on a web map with filters, histogram, Top-N, and time series.
>>>>>>> e82da24782ccc491b7db0860f9c5120db1204efd
