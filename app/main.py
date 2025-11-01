# main.py
# Bokeh server app for Rainfall_Data_LL.csv (SUBDIVISION, YEAR, JAN..DEC, Latitude, Longitude)
# Bokeh 3.8+ compatible

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import (
    Button,
    ColumnDataSource,
    ColorBar,
    BasicTicker,
    DataRange1d,
    DateRangeSlider,
    Div,
    HoverTool,
    LinearColorMapper,
    MultiChoice,
    PrintfTickFormatter,
    Select,
    Slider,
    TapTool,
    TextInput,
    MercatorTicker,
    MercatorTickFormatter,
    DatetimeTickFormatter,
    NumeralTickFormatter,
)
from bokeh.palettes import Viridis256, Inferno256, Turbo256
from bokeh.plotting import figure
from bokeh.themes import Theme
import xyzservices.providers as xyz

# ---------------------------
# Paths / theme
# ---------------------------

HERE = Path(__file__).parent
DEFAULT_DATA_PATHS = [
    HERE / "data" / "Rainfall_Data_LL.csv",
    HERE / "Rainfall_Data_LL.csv",
]

theme_file = HERE / "theme.yaml"
if theme_file.exists():
    curdoc().theme = Theme(filename=str(theme_file))
else:
    curdoc().theme = "dark_minimal"

curdoc().title = "Rainfall Explorer"

# ---------------------------
# Helpers
# ---------------------------

def _maybe_path() -> Path:
    for p in DEFAULT_DATA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Rainfall CSV not found. Expected at ./data/Rainfall_Data_LL.csv "
        "or ./Rainfall_Data_LL.csv relative to main.py."
    )

def lonlat_to_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = 6378137.0
    x = np.radians(lon) * k
    lat = np.clip(lat, -85.05112878, 85.05112878)
    y = np.log(np.tan((np.pi / 4.0) + (np.radians(lat) / 2.0))) * k
    return x, y

def padded_range(vals: np.ndarray, pad_ratio: float = 0.05) -> tuple[float, float]:
    if len(vals) == 0:
        return -1.0, 1.0
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    span = vmax - vmin
    if span <= 0:
        span = abs(vmin) if vmin != 0 else 1.0
    pad = span * pad_ratio
    return vmin - pad, vmax + pad

# ---------------------------
# Load & transform YOUR dataset
# ---------------------------

csv_path = _maybe_path()
DF0 = pd.read_csv(csv_path)

# Expected columns:
# 'SUBDIVISION', 'YEAR', monthly 'JAN'..'DEC', 'Latitude', 'Longitude'
month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
month_map = {m:i+1 for i,m in enumerate(month_order)}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

# Identify month columns present (case-insensitive)
months_present = []
for m in month_order:
    for c in DF0.columns:
        if _norm(c) == _norm(m):
            months_present.append(c)
            break
if len(months_present) == 0:
    raise ValueError("No monthly columns found. Expected JAN..DEC in the CSV.")

# Identify required id vars
subdiv_col = next((c for c in ["SUBDIVISION","Subdivision","subdivision","REGION","District","Zone","Province","State"] if c in DF0.columns), None)
if subdiv_col is None:
    raise ValueError("SUBDIVISION column not found.")

year_col = next((c for c in ["YEAR","Year","year","YR","Yr"] if c in DF0.columns), None)
if year_col is None:
    raise ValueError("YEAR column not found.")

lat_col = next((c for c in ["Latitude","LATITUDE","lat","Lat","Lat_DD"] if c in DF0.columns), None)
if lat_col is None:
    raise ValueError("Latitude column not found.")

lon_col = next((c for c in ["Longitude","LONGITUDE","lon","Lon","Lng","Long_DD"] if c in DF0.columns), None)
if lon_col is None:
    raise ValueError("Longitude column not found.")

# Melt to long format
melted = DF0.melt(
    id_vars=[subdiv_col, year_col, lat_col, lon_col],
    value_vars=months_present,
    var_name="month_label",
    value_name="rain",
)

# Month index (1..12) and date
melted["month_label_norm"] = melted["month_label"].str.upper().str.slice(0,3)
melted["month"] = melted["month_label_norm"].map(month_map)

melted["date"] = pd.to_datetime(
    dict(year=melted[year_col].astype(int), month=melted["month"].astype(int), day=1),
    errors="coerce",
).dt.tz_localize(None)

# Rename for the app
melted = melted.rename(columns={
    subdiv_col: "subdivision",
    year_col: "year",
    lat_col: "lat",
    lon_col: "lon",
})
melted["station"] = melted["subdivision"].astype(str)  # use subdivision as label
melted["rain"] = pd.to_numeric(melted["rain"], errors="coerce")

# Drop NAs and invalid
DF_ALL = melted.dropna(subset=["lon","lat","rain","date"]).copy()

# Web Mercator
mx, my = lonlat_to_mercator(DF_ALL["lon"].to_numpy(), DF_ALL["lat"].to_numpy())
DF_ALL["mx"] = mx
DF_ALL["my"] = my

# Bounds & choices
DATE_MIN = DF_ALL["date"].min()
DATE_MAX = DF_ALL["date"].max()
py_date_min = pd.to_datetime(DATE_MIN).to_pydatetime()
py_date_max = pd.to_datetime(DATE_MAX).to_pydatetime()

SUBDIVS = sorted([s for s in DF_ALL["subdivision"].dropna().unique().tolist() if s.strip() != ""])
DEFAULT_SUBDIVS = SUBDIVS[:]

# ---------------------------
# Widgets
# ---------------------------

palette_select = Select(title="Color Palette", value="Viridis256",
                        options=["Viridis256", "Inferno256", "Turbo256"])

subdivision_filter = MultiChoice(title="Subdivisions", options=SUBDIVS, value=DEFAULT_SUBDIVS)

station_search = TextInput(title="Filter by Station/Name", placeholder="type to filter (optional)")

topn_slider = Slider(title="Top N stations (by total rain)", start=5, end=50, step=1, value=10)

date_slider = DateRangeSlider(
    title="Time Range",
    value=(py_date_min, py_date_max),  # Python datetimes prevent snap-back
    start=py_date_min,
    end=py_date_max,
    step=24 * 60 * 60 * 1000,  # daily step (ms)
)

reset_btn = Button(label="Reset filters", button_type="default")

hint_div = Div(
    text="<i>Tip:</i> Drag the time range, select subdivisions, or search stations. Tap a point to see its time series.",
    sizing_mode="stretch_width",
)

# ---------------------------
# Data sources
# ---------------------------

source_map = ColumnDataSource(data=dict(mx=[], my=[], station=[], subdivision=[], date=[], rain=[]))
source_hist = ColumnDataSource(data=dict(top=[], left=[], right=[]))
source_bar = ColumnDataSource(data=dict(station=[], total=[]))
source_ts = ColumnDataSource(data=dict(date=[], rain=[]))

# ---------------------------
# Color mapping
# ---------------------------

def current_palette() -> List[str]:
    if palette_select.value == "Inferno256":
        return list(Inferno256)
    if palette_select.value == "Turbo256":
        return list(Turbo256)
    return list(Viridis256)

color_mapper = LinearColorMapper(palette=current_palette(), low=0.0, high=1.0)

# ---------------------------
# Figures (with correct axis labels/formatters)
# ---------------------------

x_min, x_max = padded_range(DF_ALL["mx"].to_numpy())
y_min, y_max = padded_range(DF_ALL["my"].to_numpy())

# MAP (Web Mercator; show lon/lat in degrees)
p_map = figure(
    title="Rainfall map",
    x_axis_type="mercator",
    y_axis_type="mercator",
    x_range=DataRange1d(start=x_min, end=x_max),
    y_range=DataRange1d(start=y_min, end=y_max),
    height=520,
    sizing_mode="stretch_both",
    tools="pan,wheel_zoom,reset,save,tap",
    active_scroll="wheel_zoom",
)
p_map.add_tile(xyz.CartoDB.DarkMatter, retina=True)

# Format mercator axes in degrees
p_map.xaxis.ticker = MercatorTicker(dimension="lon")
p_map.yaxis.ticker = MercatorTicker(dimension="lat")
p_map.xaxis.formatter = MercatorTickFormatter(dimension="lon")
p_map.yaxis.formatter = MercatorTickFormatter(dimension="lat")
p_map.xaxis.axis_label = "Longitude (°)"
p_map.yaxis.axis_label = "Latitude (°)"

r_map = p_map.circle(
    x="mx",
    y="my",
    size=8,
    source=source_map,
    fill_color={"field": "rain", "transform": color_mapper},
    line_color=None,
    fill_alpha=0.9,
)

p_map.add_tools(
    HoverTool(
        tooltips=[("Station", "@station"),
                  ("Subdivision", "@subdivision"),
                  ("Date", "@date{%Y-%m}"),
                  ("Rain", "@rain{0.00}")],
        formatters={"@date": "datetime"},
        renderers=[r_map],
    ),
    TapTool(),
)

color_bar = ColorBar(
    color_mapper=color_mapper,
    ticker=BasicTicker(desired_num_ticks=8),
    formatter=PrintfTickFormatter(format="%0.2f"),
    label_standoff=12,
    location=(0, 0),
)
p_map.add_layout(color_bar, "right")

# HISTOGRAM
p_hist = figure(
    title="Rainfall distribution (filtered)",
    height=300,  # taller for visibility
    sizing_mode="stretch_both",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
p_hist.quad(top="top", bottom=0, left="left", right="right",
            source=source_hist, line_color=None, fill_alpha=0.8)
p_hist.xaxis.axis_label = "Monthly Rain (mm)"
p_hist.yaxis.axis_label = "Frequency"
p_hist.xaxis.formatter = NumeralTickFormatter(format="0.0")

# TOP-N BAR
p_bar = figure(
    title="Top-N stations by total rainfall (filtered)",
    height=300,  # taller for visibility
    x_range=[],
    sizing_mode="stretch_both",
    tools="pan,xwheel_zoom,box_zoom,reset,save",
)
p_bar.vbar(x="station", top="total", width=0.8, source=source_bar)
p_bar.xaxis.major_label_orientation = math.pi / 3
p_bar.xaxis.axis_label = "Subdivision / Station"
p_bar.yaxis.axis_label = "Total Rain (mm)"
p_bar.yaxis.formatter = NumeralTickFormatter(format="0.0")
p_bar.min_border_bottom = 60  # room for long labels

# TIME SERIES
p_ts = figure(
    title="Time Series (tap a dot on the map)",
    height=280,
    x_axis_type="datetime",
    sizing_mode="stretch_both",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
p_ts.line(x="date", y="rain", source=source_ts)
p_ts.circle(x="date", y="rain", source=source_ts, size=5)
p_ts.xaxis.axis_label = "Month"
p_ts.yaxis.axis_label = "Rain (mm)"
p_ts.xaxis.formatter = DatetimeTickFormatter(
    months="%Y-%m",
    years="%Y",
    days="%Y-%m-%d",
)
p_ts.yaxis.formatter = NumeralTickFormatter(format="0.0")

# --- visual breathing room ---
p_map.margin = (0, 0, 20, 0)       # space below map
p_hist.margin = (10, 10, 0, 0)
p_bar.margin  = (10, 0, 0, 10)
p_ts.margin   = (12, 0, 0, 0)

# global minor style tweaks
for f in (p_hist, p_bar, p_ts, p_map):
    f.grid.grid_line_alpha = 0.25
for f in (p_hist, p_bar, p_ts):
    f.title.text_font_size = "14pt"
    f.xaxis.axis_label_text_font_size = "12pt"
    f.yaxis.axis_label_text_font_size = "12pt"

# ---------------------------
# Filtering & updates
# ---------------------------

def _as_ts(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return pd.to_datetime(v, unit="ms")
    return pd.to_datetime(v)

def filtered_df() -> pd.DataFrame:
    v0, v1 = date_slider.value
    d0, d1 = _as_ts(v0), _as_ts(v1)
    if getattr(d0, "tzinfo", None): d0 = d0.tz_localize(None)
    if getattr(d1, "tzinfo", None): d1 = d1.tz_localize(None)

    mask = (DF_ALL["date"] >= d0) & (DF_ALL["date"] <= d1)

    chosen = subdivision_filter.value or []
    if chosen:
        mask &= DF_ALL["subdivision"].isin(chosen)

    txt = station_search.value.strip().lower()
    if txt:
        mask &= DF_ALL["station"].str.lower().str.contains(txt, na=False)

    return DF_ALL.loc[mask].copy()

def recompute_hist(df: pd.DataFrame, bins: int = 30) -> dict:
    if df.empty:
        return dict(top=[], left=[], right=[])
    hist, edges = np.histogram(df["rain"].to_numpy(), bins=bins)
    return dict(top=hist, left=edges[:-1], right=edges[1:])

def recompute_topn(df: pd.DataFrame, n: int) -> tuple[List[str], List[float]]:
    if df.empty:
        return [], []
    g = df.groupby("station", dropna=False)["rain"].sum().sort_values(ascending=False).head(int(n))
    return g.index.tolist(), g.values.tolist()

def refresh_all():
    df = filtered_df()

    if df.empty:
        source_map.data = dict(mx=[], my=[], station=[], subdivision=[], date=[], rain=[])
        source_hist.data = recompute_hist(df)
        source_bar.data = dict(station=[], total=[])
        source_ts.data = dict(date=[], rain=[])
        p_ts.title.text = "Time Series (tap a dot on the map)"
        color_mapper.low, color_mapper.high = 0.0, 1.0
    else:
        rvals = df["rain"].to_numpy()
        rmin = float(np.nanmin(rvals))
        rmax = float(np.nanmax(rvals))
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin == rmax:
            rmin, rmax = 0.0, max(1.0, float(rmax) if np.isfinite(rmax) else 1.0)
        color_mapper.low, color_mapper.high = rmin, rmax

        source_map.data = dict(
            mx=df["mx"].to_numpy(),
            my=df["my"].to_numpy(),
            station=df["station"].astype(str).tolist(),
            subdivision=df["subdivision"].astype(str).tolist(),
            date=df["date"].to_numpy(),
            rain=df["rain"].to_numpy(),
        )
        source_hist.data = recompute_hist(df)
        labs, vals = recompute_topn(df, topn_slider.value)
        source_bar.data = dict(station=labs, total=vals)
        p_bar.x_range.factors = labs

    color_mapper.palette = current_palette()

def update_timeseries(attr, old, new):
    inds = r_map.data_source.selected.indices
    if not inds:
        source_ts.data = dict(date=[], rain=[])
        p_ts.title.text = "Time Series (tap a dot on the map)"
        return

    idx = inds[0]
    try:
        st = source_map.data["station"][idx]
    except Exception:
        source_ts.data = dict(date=[], rain=[])
        p_ts.title.text = "Time Series (tap a dot on the map)"
        return

    v0, v1 = date_slider.value
    d0, d1 = _as_ts(v0), _as_ts(v1)
    if getattr(d0, "tzinfo", None): d0 = d0.tz_localize(None)
    if getattr(d1, "tzinfo", None): d1 = d1.tz_localize(None)

    df_st = DF_ALL[(DF_ALL["station"] == st) & (DF_ALL["date"] >= d0) & (DF_ALL["date"] <= d1)].copy()
    if df_st.empty:
        source_ts.data = dict(date=[], rain=[])
        p_ts.title.text = f"Time Series — {st} (no data in range)"
        return

    df_st.sort_values("date", inplace=True, kind="stable")
    source_ts.data = dict(date=df_st["date"].to_numpy(), rain=df_st["rain"].to_numpy())
    p_ts.title.text = f"Time Series — {st}"

# ---------------------------
# Events
# ---------------------------

def make_cb(widget):
    def _cb(attr, old, new):
        if widget is palette_select:
            color_mapper.palette = current_palette()
        refresh_all()
    return _cb

palette_select.on_change("value", make_cb(palette_select))
subdivision_filter.on_change("value", make_cb(subdivision_filter))
station_search.on_change("value", make_cb(station_search))
topn_slider.on_change("value", make_cb(topn_slider))

# critical: throttled to avoid mid-drag flicker/snap-back
date_slider.on_change("value_throttled", make_cb(date_slider))

r_map.data_source.selected.on_change("indices", update_timeseries)

def do_reset():
    subdivision_filter.value = DEFAULT_SUBDIVS[:]
    station_search.value = ""
    topn_slider.value = 10
    date_slider.value = (py_date_min, py_date_max)
    r_map.data_source.selected.indices = []
    refresh_all()
    p_ts.title.text = "Time Series (tap a dot on the map)"

reset_btn.on_click(do_reset)

# ---------------------------
# Initial draw & layout
# ---------------------------

refresh_all()

controls_left = column(
    palette_select,
    date_slider,
    subdivision_filter,
    station_search,
    topn_slider,
    row(reset_btn, Spacer(width=12)),
    hint_div,
    width=360,
    sizing_mode="stretch_height",
)

right_col = column(
    p_map,
    Spacer(height=24),  # push the lower row down
    row(p_hist, p_bar, sizing_mode="stretch_both"),
    Spacer(height=16),  # small gap before TS
    p_ts,
    sizing_mode="stretch_both",
)

curdoc().add_root(row(controls_left, right_col, sizing_mode="stretch_both"))
