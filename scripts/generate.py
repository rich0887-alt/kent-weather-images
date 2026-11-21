#!/usr/bin/env python3
"""Kent Weather Facebook graphics pipeline.

Generate daily, 7-day and 10-day forecast graphics from live Open-Meteo,
NOAA GFS and NASA GIBS data, publish them to the public GitHub repo, and print a JSON
summary of the public raw URLs and the figures used in the graphics.

Usage:
  /workspace/kent-weather/.venv/bin/python /workspace/kent-weather/generate.py daily
  /workspace/kent-weather/.venv/bin/python /workspace/kent-weather/generate.py 7day
  /workspace/kent-weather/.venv/bin/python /workspace/kent-weather/generate.py 7day --start YYYY-MM-DD
  /workspace/kent-weather/.venv/bin/python /workspace/kent-weather/generate.py 10day

Optional flags:
  --daily-image {gfs,satellite,both,none}
                Second image on the daily post. Default gfs = NOAA GFS model
                chart (gfs_chart.png: MSLP isobars + precipitation rate at 12:00
                UK time); if GFS fails it falls back to the NASA satellite image,
                and if that fails too the post has today_chart.png only (see
                "warnings" in the JSON).
  --gfs-loop    Daily mode: also build gfs_loop.mp4 (+ gfs_loop.gif when under
                8 MB), hourly GFS frames 12:00-16:00 UK time. Off by default;
                a failure only adds a warning.
  --date YYYY-MM-DD
                Daily mode only: build the daily graphics for another date
  --repo PATH   Path to the kent-weather-images git repo
  --no-push     Generate files but do not commit or push
  --no-prune    Do not delete posts/ folders older than 30 days
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Bootstrap: if third-party deps are missing, create/use a local venv and re-exec
# ---------------------------------------------------------------------------

REQUIRED_MODULES = ("matplotlib", "numpy", "PIL", "scipy", "shapely", "requests")
VENV_DIR = Path("/workspace/kent-weather/.venv")
PIP_PACKAGES = ["matplotlib", "numpy", "pillow", "scipy", "shapely", "requests"]


def _missing_modules() -> list[str]:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def bootstrap_deps() -> None:
    if not _missing_modules():
        return
    py = VENV_DIR / "bin" / "python"
    if Path(sys.executable).resolve() != py.resolve():
        if not py.exists():
            log(f"Creating virtualenv at {VENV_DIR}")
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        pip = VENV_DIR / "bin" / "pip"
        req = Path(__file__).resolve().parent / "requirements.txt"
        log("Installing Python dependencies into the virtualenv…")
        cmd = [str(pip), "install", "--disable-pip-version-check"]
        if req.exists():
            cmd += ["-r", str(req)]
        else:
            cmd += PIP_PACKAGES
        subprocess.run(cmd, check=True)
        os.execv(str(py), [str(py), *sys.argv])
    # Already running inside the venv but still missing modules
    pip = Path(sys.executable).resolve().parent / "pip"
    req = Path(__file__).resolve().parent / "requirements.txt"
    log("Installing missing Python dependencies…")
    cmd = [str(pip), "install", "--disable-pip-version-check"]
    if req.exists():
        cmd += ["-r", str(req)]
    else:
        cmd += PIP_PACKAGES
    subprocess.run(cmd, check=True)
    still = _missing_modules()
    if still:
        raise SystemExit(f"Failed to import required modules after install: {still}")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# Must run before importing matplotlib so a bare `python3 generate.py` can self-heal.
if __name__ == "__main__":
    bootstrap_deps()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import font_manager, patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import griddata
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape

mpl.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TZ = ZoneInfo("Europe/London")
REPO_DEFAULT = Path("/workspace/kent-weather-images")
RAW_BASE = "https://raw.githubusercontent.com/rich0887-alt/kent-weather-images/main"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
USER_AGENT = "KentWeatherGraphics/1.0 (Open-Meteo + NASA GIBS; Facebook page graphics)"

FONT_DIR = Path("/usr/share/fonts/truetype/sand-box/google/Poppins")
FONTS = {
    "regular": FONT_DIR / "Poppins-Regular.ttf",
    "medium": FONT_DIR / "Poppins-Medium.ttf",
    "semibold": FONT_DIR / "Poppins-SemiBold.ttf",
    "bold": FONT_DIR / "Poppins-Bold.ttf",
    "light": FONT_DIR / "Poppins-Light.ttf",
}

NAVY = "#14344A"
NAVY_RGB = (20, 52, 74)
NAVY_DEEP = "#0E2433"
INK = "#1A2B3C"
INK_RGB = (26, 43, 60)
MUTED = "#5B6C7A"
MUTED_RGB = (91, 108, 122)
SOFT = "#EEF3F8"
SOFT_RGB = (238, 243, 248)
CARD = "#FFFFFF"
HIGH_C = (200, 90, 23)
LOW_C = (31, 106, 154)
RAIN_C = (44, 123, 182)
TEAL = "#1F7A78"
TEAL_RGB = (31, 122, 120)
LINE = "#D5DEE7"
SEA = "#D4E4EE"
LAND = "#F3F0E8"

TOWNS = [
    # name, lon, lat, ha, va, dx, dy (label offset in degrees, map only)
    ("Ashford", 0.8750, 51.1465, "center", "top", 0.00, -0.035),
    ("Canterbury", 1.0789, 51.2802, "right", "bottom", -0.04, 0.025),
    ("Maidstone", 0.5227, 51.2704, "right", "bottom", -0.03, 0.022),
    ("Tunbridge Wells", 0.2630, 51.1320, "left", "top", 0.04, -0.030),
    ("Folkestone", 1.1694, 51.0814, "center", "top", 0.00, -0.032),
    ("Margate", 1.3862, 51.3813, "right", "bottom", -0.03, 0.022),
]
TOWN_NAMES = [t[0] for t in TOWNS]
TOWN_LATS = [t[2] for t in TOWNS]
TOWN_LONS = [t[1] for t in TOWNS]

# Animation / interpolation grid (matches the original sample)
GRID_LATS = [50.85, 50.95, 51.05, 51.15, 51.25, 51.35, 51.45, 51.55]
GRID_LONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
MAP_LON0, MAP_LON1 = 0.00, 1.60
MAP_LAT0, MAP_LAT1 = 50.85, 51.55

# GIBS true-colour layers, preferred first (Terra matches the original sample)
GIBS_LAYERS = [
    ("MODIS_Terra_CorrectedReflectance_TrueColor", "MODIS Terra true colour"),
    ("MODIS_Aqua_CorrectedReflectance_TrueColor", "MODIS Aqua true colour"),
    ("VIIRS_SNPP_CorrectedReflectance_TrueColor", "VIIRS Suomi NPP true colour"),
    ("VIIRS_NOAA20_CorrectedReflectance_TrueColor", "VIIRS NOAA-20 true colour"),
    ("VIIRS_NOAA21_CorrectedReflectance_TrueColor", "VIIRS NOAA-21 true colour"),
]
# Fetch a slightly wider box, then crop like the sample
GIBS_FETCH_LAT = (49.8, 60.2)
GIBS_FETCH_LON = (-10.8, 2.2)
GIBS_CROP_LAT = (49.85, 59.45)
GIBS_CROP_LON = (-8.7, 2.05)
GIBS_FETCH_SIZE = (1560, 1250)

IMPACT = {
    0: 0, 1: 1, 2: 2, 3: 3,
    45: 6, 48: 7,
    51: 10, 53: 11, 55: 12, 56: 13, 57: 14,
    61: 20, 63: 21, 65: 22, 66: 23, 67: 24,
    71: 30, 73: 31, 75: 32, 77: 29,
    80: 20, 81: 21, 82: 25,
    85: 30, 86: 33,
    95: 40, 96: 41, 99: 42,
}

LABELS = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Cloudy",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Heavy showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}

ACCENT = {
    "rain": (44, 123, 182),
    "drizzle": (70, 150, 190),
    "cloudy": (140, 160, 176),
    "partly": (230, 170, 70),
    "clear": (240, 180, 50),
    "snow": (160, 190, 220),
    "fog": (170, 180, 190),
    "thunder": (120, 90, 160),
}

PERIODS = [
    ("Morning", 6, 12),
    ("Afternoon", 12, 18),
    ("Evening", 18, 24),
]


class PipelineError(Exception):
    """Fatal, user-facing pipeline error."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def fp(weight: str = "regular", size: float = 12):
    return font_manager.FontProperties(fname=str(FONTS[weight]), size=size)


def pil_font(weight: str = "regular", size: int = 32) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS[weight]), size=size)


def round_temp(x: float) -> int:
    """Traditional half-up rounding to whole °C (values may be negative)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise PipelineError("Missing temperature value in Open-Meteo response")
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def round_num(x: float) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise PipelineError("Missing numeric value in Open-Meteo response")
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def pick_code(codes: list[int]) -> int:
    codes = [int(c) for c in codes if c is not None]
    if not codes:
        raise PipelineError("No weather codes available to summarise")
    counts = Counter(codes)
    max_n = max(counts.values())
    tied = [c for c, n in counts.items() if n == max_n]
    return max(tied, key=lambda c: IMPACT.get(c, 0))


def icon_kind(code: int) -> str:
    if code == 0:
        return "clear"
    if code in (1, 2):
        return "partly"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "thunder"
    return "cloudy"


def today_london() -> date:
    return datetime.now(TZ).date()


def parse_iso_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise PipelineError(f"Invalid date {s!r}; use YYYY-MM-DD") from exc


def daterange(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def format_long_date(d: date) -> str:
    return d.strftime("%A %-d %B %Y")


def format_range(start: date, end: date) -> str:
    if start.year == end.year and start.month == end.month:
        return f"{start.strftime('%A %-d')} – {end.strftime('%A %-d %B %Y')}"
    if start.year == end.year:
        return f"{start.strftime('%A %-d %B')} – {end.strftime('%A %-d %B %Y')}"
    return f"{start.strftime('%A %-d %B %Y')} – {end.strftime('%A %-d %B %Y')}"


def require_val(value, label: str):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise PipelineError(f"Missing {label} in Open-Meteo response")
    return value


def mean(xs: list[float]) -> float:
    if not xs:
        raise PipelineError("Cannot average an empty list of forecast values")
    return sum(xs) / len(xs)


def public_url(rel: str) -> str:
    return f"{RAW_BASE}/{rel.lstrip('/')}"


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def check_fonts() -> None:
    missing = [p.name for p in FONTS.values() if not p.exists()]
    if missing:
        raise PipelineError(
            f"Required Poppins fonts not found in {FONT_DIR}: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_get(url: str, *, timeout: int = 30, tries: int = 4, stream: bool = False) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
                stream=stream,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = PipelineError(f"HTTP {resp.status_code} from {url}")
                time.sleep(min(2 ** attempt, 8))
                continue
            if resp.status_code >= 400:
                raise PipelineError(f"HTTP {resp.status_code} from {url}")
            return resp
        except requests.RequestException as exc:
            last_err = exc
            log(f"  request failed ({attempt}/{tries}): {exc}")
            time.sleep(min(2 ** attempt, 8))
    raise PipelineError(f"Request failed after {tries} tries: {url} ({last_err})")


def http_get_json(url: str, **kw) -> Any:
    resp = http_get(url, **kw)
    try:
        return resp.json()
    except ValueError as exc:
        raise PipelineError(f"Non-JSON response from {url}") from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def default_repo(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not (p / ".git").exists():
            raise PipelineError(f"--repo {p} is not a git repository")
        return p
    env = os.environ.get("KENT_WEATHER_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / ".git").exists():
            raise PipelineError(f"KENT_WEATHER_REPO {p} is not a git repository")
        return p
    here = Path(__file__).resolve().parent
    if here.name == "scripts" and (here.parent / ".git").exists():
        return here.parent
    if (REPO_DEFAULT / ".git").exists():
        return REPO_DEFAULT.resolve()
    raise PipelineError(
        "Cannot find the kent-weather-images git repo. Pass --repo PATH."
    )


def data_dir(repo: Path) -> Path:
    candidates = [
        repo / "scripts" / "data",
        Path(__file__).resolve().parent / "data",
        Path("/workspace/kent-weather-samples/data"),
    ]
    for c in candidates:
        if (c / "kent_ceremonial_osm.geojson").exists():
            return c
    raise PipelineError(
        "Map data (kent_ceremonial_osm.geojson) not found next to the script or in the repo."
    )


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

DAILY_VARS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]
HOURLY_TOWN_VARS = [
    "weather_code",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
]
HOURLY_GRID_VARS = ["temperature_2m", "precipitation", "precipitation_probability"]


def _as_location_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "latitude" in payload:
        return [payload]
    raise PipelineError("Unexpected Open-Meteo response shape")


def fetch_open_meteo(
    lats: list[float],
    lons: list[float],
    *,
    daily: list[str] | None = None,
    hourly: list[str] | None = None,
    forecast_days: int = 16,
    batch_size: int = 20,
) -> list[dict]:
    """Fetch one Open-Meteo payload per coordinate, batched to keep URLs short."""
    if len(lats) != len(lons):
        raise PipelineError("Latitude/longitude lists are different lengths")
    out: list[dict | None] = [None] * len(lats)
    for start in range(0, len(lats), batch_size):
        sl_lats = lats[start : start + batch_size]
        sl_lons = lons[start : start + batch_size]
        params = [
            ("latitude", ",".join(f"{v:.5f}" for v in sl_lats)),
            ("longitude", ",".join(f"{v:.5f}" for v in sl_lons)),
            ("timezone", "Europe/London"),
            ("wind_speed_unit", "mph"),
            ("forecast_days", str(forecast_days)),
        ]
        if daily:
            params.append(("daily", ",".join(daily)))
        if hourly:
            params.append(("hourly", ",".join(hourly)))
        # Build URL ourselves so commas in coordinate lists stay unescaped as Open-Meteo expects
        from urllib.parse import urlencode

        url = f"{OPEN_METEO}?{urlencode(params, safe=',')}"
        log(f"  Open-Meteo batch {start // batch_size + 1}: {len(sl_lats)} location(s)")
        payload = http_get_json(url, timeout=60, tries=4)
        locs = _as_location_list(payload)
        if len(locs) != len(sl_lats):
            raise PipelineError(
                f"Open-Meteo returned {len(locs)} locations, expected {len(sl_lats)}"
            )
        for i, loc in enumerate(locs):
            if loc.get("error"):
                raise PipelineError(f"Open-Meteo error: {loc.get('reason', loc['error'])}")
            out[start + i] = loc
    return out  # type: ignore[return-value]


def index_daily(loc: dict, day: date) -> int:
    times = loc["daily"]["time"]
    key = day.isoformat()
    try:
        return times.index(key)
    except ValueError as exc:
        raise PipelineError(
            f"Open-Meteo daily series does not include {key} "
            f"(got {times[0] if times else 'no days'}–{times[-1] if times else ''})"
        ) from exc


def town_daily_row(loc: dict, day: date) -> dict[str, Any]:
    i = index_daily(loc, day)
    d = loc["daily"]
    code = int(require_val(d["weather_code"][i], "weather_code"))
    tmax = float(require_val(d["temperature_2m_max"][i], "temperature_2m_max"))
    tmin = float(require_val(d["temperature_2m_min"][i], "temperature_2m_min"))
    pop = float(require_val(d["precipitation_probability_max"][i], "precipitation_probability_max"))
    precip = float(require_val(d["precipitation_sum"][i], "precipitation_sum"))
    wind = float(require_val(d["wind_speed_10m_max"][i], "wind_speed_10m_max"))
    gusts = float(require_val(d["wind_gusts_10m_max"][i], "wind_gusts_10m_max"))
    return {
        "date": day.isoformat(),
        "code": code,
        "label": LABELS.get(code, "—"),
        "tmax": tmax,
        "tmin": tmin,
        "tmax_r": round_temp(tmax),
        "tmin_r": round_temp(tmin),
        "pop": pop,
        "pop_r": round_num(pop),
        "precip": precip,
        "wind": wind,
        "gusts": gusts,
        "wind_r": round_num(wind),
        "gusts_r": round_num(gusts),
    }


def average_daily(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        raise PipelineError("No town rows to average")
    code = pick_code([r["code"] for r in rows])
    tmax = mean([r["tmax"] for r in rows])
    tmin = mean([r["tmin"] for r in rows])
    pop = mean([r["pop"] for r in rows])
    precip = mean([r["precip"] for r in rows])
    wind = mean([r["wind"] for r in rows])
    gusts = mean([r["gusts"] for r in rows])
    return {
        "date": rows[0]["date"],
        "code": code,
        "label": LABELS.get(code, "—"),
        "tmax": tmax,
        "tmin": tmin,
        "tmax_r": round_temp(tmax),
        "tmin_r": round_temp(tmin),
        "pop": pop,
        "pop_r": round_num(pop),
        "precip": precip,
        "wind": wind,
        "gusts": gusts,
        "wind_r": round_num(wind),
        "gusts_r": round_num(gusts),
        "codes": [r["code"] for r in rows],
        "tmaxs": [r["tmax"] for r in rows],
        "tmins": [r["tmin"] for r in rows],
    }


def hourly_by_date(loc: dict) -> dict[str, list[dict]]:
    h = loc["hourly"]
    times = h["time"]
    out: dict[str, list[dict]] = {}
    for i, ts in enumerate(times):
        # Open-Meteo hourly times are 'YYYY-MM-DDTHH:MM' in the requested timezone
        day, hm = ts.split("T")
        hour = int(hm.split(":")[0])
        rec = {"time": ts, "hour": hour}
        for key in h:
            if key == "time":
                continue
            rec[key] = h[key][i]
        out.setdefault(day, []).append(rec)
    return out


def period_summary(town_hourlies: list[dict[str, list[dict]]], day: date, start_h: int, end_h: int) -> dict[str, Any]:
    codes: list[int] = []
    temps: list[float] = []
    pops: list[float] = []
    precips: list[float] = []
    key = day.isoformat()
    for by_day in town_hourlies:
        hours = [row for row in by_day.get(key, []) if start_h <= row["hour"] < end_h]
        if not hours:
            continue
        for row in hours:
            if row.get("weather_code") is not None:
                codes.append(int(row["weather_code"]))
            if row.get("temperature_2m") is not None:
                temps.append(float(row["temperature_2m"]))
            if row.get("precipitation_probability") is not None:
                pops.append(float(row["precipitation_probability"]))
            if row.get("precipitation") is not None:
                precips.append(float(row["precipitation"]))
    if not temps or not codes:
        raise PipelineError(
            f"No hourly Open-Meteo data for {key} between {start_h:02d}:00 and {end_h:02d}:00"
        )
    code = pick_code(codes)
    return {
        "code": code,
        "label": LABELS.get(code, "—"),
        "temp": mean(temps),
        "temp_r": round_temp(mean(temps)),
        "tmin_r": round_temp(min(temps)),
        "tmax_r": round_temp(max(temps)),
        "pop": max(pops) if pops else 0.0,
        "pop_r": round_num(max(pops) if pops else 0.0),
        "precip": sum(precips) / len(town_hourlies) if precips else 0.0,
    }


def fetch_towns(forecast_days: int = 16) -> list[dict]:
    log("Fetching Open-Meteo daily/hourly data for the six Kent towns…")
    return fetch_open_meteo(
        TOWN_LATS,
        TOWN_LONS,
        daily=DAILY_VARS,
        hourly=HOURLY_TOWN_VARS,
        forecast_days=forecast_days,
        batch_size=6,
    )


def fetch_grid(forecast_days: int = 16) -> dict:
    log("Fetching Open-Meteo hourly grid over Kent…")
    lats: list[float] = []
    lons: list[float] = []
    for lat in GRID_LATS:
        for lon in GRID_LONS:
            lats.append(lat)
            lons.append(lon)
    locs = fetch_open_meteo(
        lats,
        lons,
        hourly=HOURLY_GRID_VARS,
        forecast_days=forecast_days,
        batch_size=18,
    )
    times = locs[0]["hourly"]["time"]
    # Every location must share the same time axis
    for loc in locs[1:]:
        if loc["hourly"]["time"] != times:
            raise PipelineError("Open-Meteo grid locations have mismatched hourly time axes")
    return {
        "times": times,
        "lons": np.array([loc["longitude"] for loc in locs], dtype=float),
        "lats": np.array([loc["latitude"] for loc in locs], dtype=float),
        "precip": np.array([loc["hourly"]["precipitation"] for loc in locs], dtype=float),
        "temp": np.array([loc["hourly"]["temperature_2m"] for loc in locs], dtype=float),
        "req_lats": GRID_LATS,
        "req_lons": GRID_LONS,
    }


# ---------------------------------------------------------------------------
# Icons (same flat style as the original sample generator)
# ---------------------------------------------------------------------------

def make_icon(kind: str, size: int = 256) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size

    def cloud(cx, cy, sc, fill, outline=None):
        r1, r2, r3 = int(0.22 * sc), int(0.28 * sc), int(0.20 * sc)
        parts = [
            (cx - int(0.28 * sc), cy - int(0.02 * sc), r1),
            (cx + int(0.02 * sc), cy - int(0.14 * sc), r2),
            (cx + int(0.30 * sc), cy - int(0.00 * sc), r3),
        ]
        for x, y, r in parts:
            d.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline)
        body_l = cx - int(0.42 * sc)
        body_r = cx + int(0.46 * sc)
        body_t = cy - int(0.02 * sc)
        body_b = cy + int(0.28 * sc)
        d.rounded_rectangle(
            (body_l, body_t, body_r, body_b),
            radius=int(0.22 * sc),
            fill=fill,
            outline=outline,
        )

    def sun(cx, cy, r, col=(244, 180, 40, 255)):
        ray = (244, 180, 40, 255)
        for i in range(8):
            ang = math.radians(i * 45)
            inner = r * 1.25
            outer = r * 1.72
            x1, y1 = cx + inner * math.cos(ang), cy + inner * math.sin(ang)
            x2, y2 = cx + outer * math.cos(ang), cy + outer * math.sin(ang)
            d.line((x1, y1, x2, y2), fill=ray, width=max(3, s // 32))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)

    if kind == "clear":
        sun(s * 0.50, s * 0.50, int(s * 0.22))
    elif kind == "partly":
        sun(s * 0.38, s * 0.38, int(s * 0.16))
        cloud(s * 0.55, s * 0.54, s * 0.92, (230, 236, 242, 255))
    elif kind == "cloudy":
        cloud(s * 0.50, s * 0.46, s * 1.00, (168, 184, 199, 255))
        cloud(s * 0.46, s * 0.52, s * 0.92, (196, 208, 220, 255))
    elif kind == "drizzle":
        cloud(s * 0.50, s * 0.38, s * 0.92, (170, 186, 202, 255))
        for x, y in [(0.32, 0.72), (0.46, 0.80), (0.60, 0.72), (0.74, 0.80)]:
            r = int(s * 0.035)
            d.ellipse((s * x - r, s * y - r, s * x + r, s * y + r), fill=(70, 140, 200, 255))
    elif kind == "rain":
        cloud(s * 0.50, s * 0.36, s * 0.92, (140, 160, 180, 255))
        for x, y in [(0.30, 0.62), (0.44, 0.68), (0.58, 0.62), (0.72, 0.68)]:
            d.line(
                (s * x, s * y, s * x + s * 0.04, s * y + s * 0.20),
                fill=(45, 115, 190, 255),
                width=max(4, s // 28),
            )
    elif kind == "snow":
        cloud(s * 0.50, s * 0.36, s * 0.92, (180, 190, 200, 255))
        for x, y in [(0.34, 0.72), (0.50, 0.80), (0.66, 0.72)]:
            cx, cy = s * x, s * y
            for i in range(3):
                ang = math.radians(i * 60)
                d.line(
                    (
                        cx - 8 * math.cos(ang),
                        cy - 8 * math.sin(ang),
                        cx + 8 * math.cos(ang),
                        cy + 8 * math.sin(ang),
                    ),
                    fill=(230, 240, 255, 255),
                    width=3,
                )
    elif kind == "fog":
        for i, y in enumerate((0.38, 0.52, 0.66)):
            d.rounded_rectangle(
                (s * 0.18, s * y, s * 0.82, s * (y + 0.08)),
                radius=int(s * 0.04),
                fill=(170 + i * 15, 180 + i * 12, 190 + i * 10, 230),
            )
    elif kind == "thunder":
        cloud(s * 0.50, s * 0.34, s * 0.92, (120, 130, 150, 255))
        bolt = [
            (s * 0.52, s * 0.52),
            (s * 0.40, s * 0.70),
            (s * 0.50, s * 0.70),
            (s * 0.42, s * 0.90),
            (s * 0.66, s * 0.64),
            (s * 0.54, s * 0.64),
        ]
        d.polygon(bolt, fill=(244, 196, 40, 255))
    else:
        cloud(s * 0.50, s * 0.50, s * 1.00, (196, 208, 220, 255))
    return im


def icon_cache(*codes: int, size: int = 92) -> dict[str, Image.Image]:
    icons: dict[str, Image.Image] = {}
    for code in codes:
        k = icon_kind(code)
        if k not in icons:
            icons[k] = make_icon(k, 256).resize((size, size), Image.Resampling.LANCZOS)
    return icons


# ---------------------------------------------------------------------------
# Shared chart chrome
# ---------------------------------------------------------------------------

W = H = 1080


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), SOFT_RGB)
    return img, ImageDraw.Draw(img)


def draw_navy_header(draw: ImageDraw.ImageDraw, title_sub: str, dateline: str, header_h: int = 198) -> None:
    draw.rectangle((0, 0, W, header_h), fill=NAVY_RGB)
    draw.rectangle((0, header_h - 4, W, header_h), fill=TEAL_RGB)
    draw.text((56, 28), "Kent Weather", font=pil_font("bold", 54), fill=(255, 255, 255))
    draw.text((56, 96), title_sub, font=pil_font("medium", 28), fill=(200, 230, 228))
    draw.text((56, 140), dateline, font=pil_font("regular", 22), fill=(190, 210, 220))


def paste_shadow_card(img: Image.Image, box: tuple[float, float, float, float], radius: int = 16, muted: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        (box[0] + 2, box[1] + 3, box[2] + 2, box[3] + 3),
        radius=radius,
        fill=(20, 40, 60, 28),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)
    fill = (246, 247, 249) if muted else (255, 255, 255)
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    return img, draw


def draw_accent(draw: ImageDraw.ImageDraw, box, kind: str) -> None:
    col = ACCENT.get(kind, (140, 160, 176))
    ax0, ay0, ax1, ay1 = box[0] + 8, box[1] + 12, box[0] + 16, box[3] - 12
    draw.rounded_rectangle((ax0, ay0, ax1, ay1), radius=4, fill=col)


def draw_teardrop(draw: ImageDraw.ImageDraw, cx: float, cy: float, col, dsz: int = 7) -> None:
    draw.ellipse((cx - dsz, cy - dsz + 2, cx + dsz, cy + dsz + 2), fill=col)
    draw.polygon(
        [(cx, cy - 14), (cx - dsz, cy - 1), (cx + dsz, cy - 1)],
        fill=col,
    )


def draw_rain_meter(draw: ImageDraw.ImageDraw, x: float, cy: float, pop: int, font_pct, font_mm=None, precip: float | None = None, compact: bool = False) -> None:
    drop_col = (44, 123, 182) if pop >= 30 else (110, 150, 175)
    drop_cy = cy - (0 if compact else 2)
    draw_teardrop(draw, x, drop_cy, drop_col, dsz=6 if compact else 7)
    draw.text((x + 16, cy - (18 if compact else 22)), f"{pop}%", font=font_pct, fill=(30, 80, 120))
    bar_x0 = x + 16
    bar_y0 = cy + (10 if compact else 14)
    bar_w, bar_h = (90 if compact else 110), (7 if compact else 8)
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h), radius=4, fill=(226, 234, 240))
    fill_w = max(2, int(bar_w * min(pop, 100) / 100)) if pop > 0 else 0
    if fill_w:
        draw.rounded_rectangle((bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h), radius=4, fill=drop_col)
    if font_mm is not None and precip is not None and precip >= 0.5:
        draw.text((bar_x0, cy + (20 if compact else 26)), f"{precip:.1f} mm", font=font_mm, fill=MUTED_RGB)


def draw_footer(draw: ImageDraw.ImageDraw, lines: list[str], footer_h: int) -> None:
    draw.rectangle((0, H - footer_h, W, H), fill=SOFT_RGB)
    y = H - footer_h + 16
    for i, line in enumerate(lines):
        font = pil_font("semibold", 16) if i == 0 else pil_font("regular", 15)
        fill = NAVY_RGB if i == 0 else MUTED_RGB
        draw.text((56, y), line, font=font, fill=fill)
        y += 26 if i == 0 else 22


# ---------------------------------------------------------------------------
# Multi-day charts (7-day row layout, 10-day two-column)
# ---------------------------------------------------------------------------

def draw_multiday_chart(
    days: list[dict],
    *,
    title: str,
    subtitle: str,
    footer_lines: list[str],
    low_confidence_from: int | None = None,
    two_column: bool = False,
    path: Path,
) -> Path:
    img, draw = new_canvas()
    header_h = 198
    draw_navy_header(draw, title, subtitle, header_h=header_h)

    footer_h = 86 if len(footer_lines) <= 2 else 110
    top = header_h + 16
    bottom = H - footer_h
    n = len(days)
    icons = icon_cache(*(d["code"] for d in days), size=84 if two_column else 92)

    if two_column:
        cols, rows = 2, math.ceil(n / 2)
        gutter = 16
        pad_x = 28
        col_w = (W - pad_x * 2 - gutter) / 2
        row_h = (bottom - top) / rows
        font_day = pil_font("semibold", 20)
        font_date = pil_font("regular", 14)
        font_cond = pil_font("regular", 15)
        font_hi = pil_font("bold", 28)
        font_lo = pil_font("medium", 22)
        font_rain = pil_font("medium", 16)
        font_mm = pil_font("regular", 13)
        font_tag = pil_font("semibold", 11)
    else:
        cols, rows = 1, n
        pad_x = 32
        col_w = W - pad_x * 2
        gutter = 0
        row_h = (bottom - top) / rows
        font_day = pil_font("semibold", 26)
        font_date = pil_font("regular", 18)
        font_cond = pil_font("regular", 20)
        font_hi = pil_font("bold", 36)
        font_lo = pil_font("medium", 28)
        font_rain = pil_font("medium", 20)
        font_mm = pil_font("regular", 16)
        font_tag = pil_font("semibold", 12)

    for i, d in enumerate(days):
        if two_column:
            # Left column = first half, right column = second half (days 6-10 on the right)
            col = 0 if i < rows else 1
            row = i if i < rows else i - rows
            x0 = pad_x + col * (col_w + gutter)
        else:
            col, row = 0, i
            x0 = pad_x
        y0 = top + row * row_h
        box = (x0, y0 + 6, x0 + col_w, y0 + row_h - 6)
        muted = low_confidence_from is not None and i >= low_confidence_from
        img, draw = paste_shadow_card(img, box, radius=16, muted=muted)
        kind = icon_kind(d["code"])
        draw_accent(draw, box, kind)

        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        day_name = dt.strftime("%a").upper()
        date_s = dt.strftime("%-d %b")
        cy = (box[1] + box[3]) / 2
        tx = box[0] + (26 if two_column else 32)
        name_fill = (80, 96, 108) if muted else (20, 43, 60)
        date_fill = (130, 142, 152) if muted else MUTED_RGB
        draw.text((tx, cy - (22 if two_column else 28)), day_name, font=font_day, fill=name_fill)
        draw.text((tx, cy + (4 if two_column else 6)), date_s, font=font_date, fill=date_fill)

        icon = icons[kind]
        ix = int(box[0] + (118 if two_column else 230 - 32 + (box[0] - 32)))
        # Anchor icon from the card's left rather than a hard-coded page x
        ix = int(box[0] + (108 if two_column else 198))
        iy = int(cy - icon.height / 2)
        img.paste(icon, (ix, iy), icon)
        draw = ImageDraw.Draw(img)

        cond = LABELS.get(d["code"], "—")
        if two_column:
            # Condition sits under the icon only if it fits; otherwise to the right of the date.
            # With the narrower card, put a short condition between icon and temps.
            draw.text((box[0] + 196, cy - 22), cond, font=font_cond, fill=(40, 60, 75) if not muted else (110, 122, 132))
            temp_x = box[0] + 196
            draw.text((temp_x, cy + 2), f"{d['tmax_r']}°", font=font_hi, fill=HIGH_C)
            tw = draw.textlength(f"{d['tmax_r']}°", font=font_hi)
            draw.text((temp_x + tw + 8, cy + 8), f"{d['tmin_r']}°", font=font_lo, fill=LOW_C)
            rain_x = box[2] - 128
            draw_rain_meter(draw, rain_x, cy, d["pop_r"], font_rain, font_mm, d["precip"], compact=True)
            if muted:
                tag = "Lower confidence"
                tbb = draw.textbbox((0, 0), tag, font=font_tag)
                tw, th = tbb[2] - tbb[0], tbb[3] - tbb[1]
                tx1 = box[2] - 12
                ty1 = box[1] + 8
                pad_x, pad_y = 6, 3
                draw.rounded_rectangle(
                    (tx1 - tw - pad_x * 2, ty1, tx1, ty1 + th + pad_y * 2),
                    radius=6,
                    fill=(245, 236, 214),
                )
                draw.text((tx1 - tw - pad_x, ty1 + pad_y - 1), tag, font=font_tag, fill=(140, 110, 40))
        else:
            draw.text((340, cy - 16), cond, font=font_cond, fill=(40, 60, 75))
            temp_x = 640
            draw.text((temp_x, cy - 28), f"{d['tmax_r']}°", font=font_hi, fill=HIGH_C)
            draw.text((temp_x + 78, cy - 18), f"{d['tmin_r']}°", font=font_lo, fill=LOW_C)
            draw_rain_meter(draw, 840, cy, d["pop_r"], font_rain, font_mm, d["precip"], compact=False)

    draw_footer(draw, footer_lines, footer_h)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG", optimize=True)
    log(f"wrote {path} {img.size}")
    return path


# ---------------------------------------------------------------------------
# Daily 'Today in Kent' chart
# ---------------------------------------------------------------------------

def draw_daily_chart(towns: list[dict], periods: list[dict], day: date, path: Path) -> Path:
    img, draw = new_canvas()
    header_h = 178
    draw_navy_header(draw, "Today in Kent", format_long_date(day), header_h=header_h)

    footer_h = 70
    strip_h = 168
    top = header_h + 14
    # Period strip sits above the footer, town rows fill the rest
    towns_bottom = H - footer_h - strip_h - 8
    row_h = (towns_bottom - top) / 6
    pad_x = 32

    icons = icon_cache(*(t["code"] for t in towns), *(p["code"] for p in periods), size=80)
    font_town = pil_font("semibold", 22)
    font_cond = pil_font("regular", 17)
    font_hi = pil_font("bold", 32)
    font_lo = pil_font("medium", 24)
    font_rain = pil_font("medium", 18)
    font_wind = pil_font("semibold", 18)
    font_gust = pil_font("regular", 14)

    for i, t in enumerate(towns):
        y0 = top + i * row_h
        box = (pad_x, y0 + 5, W - pad_x, y0 + row_h - 5)
        img, draw = paste_shadow_card(img, box, radius=16)
        kind = icon_kind(t["code"])
        draw_accent(draw, box, kind)
        cy = (box[1] + box[3]) / 2

        name_font = font_town
        # Long names like "Tunbridge Wells" need a slightly smaller size to clear the icon
        if draw.textlength(t["name"], font=font_town) > 230:
            name_font = pil_font("semibold", 19)
        draw.text((box[0] + 30, cy - 12), t["name"], font=name_font, fill=(20, 43, 60))

        icon = icons[kind]
        ix = 280
        iy = int(cy - icon.height / 2)
        img.paste(icon, (ix, iy), icon)
        draw = ImageDraw.Draw(img)

        draw.text((368, cy - 12), t["label"], font=font_cond, fill=(40, 60, 75))

        temp_x = 560
        draw.text((temp_x, cy - 24), f"{t['tmax_r']}°", font=font_hi, fill=HIGH_C)
        draw.text((temp_x + 70, cy - 16), f"{t['tmin_r']}°", font=font_lo, fill=LOW_C)

        draw_rain_meter(draw, 740, cy, t["pop_r"], font_rain, compact=True)

        wind_x = 900
        draw.text((wind_x, cy - 20), f"{t['wind_r']} mph", font=font_wind, fill=NAVY_RGB)
        draw.text((wind_x, cy + 6), f"gusts {t['gusts_r']}", font=font_gust, fill=MUTED_RGB)

    # Morning / afternoon / evening strip
    strip_top = towns_bottom + 6
    strip_box = (pad_x, strip_top, W - pad_x, H - footer_h - 8)
    img, draw = paste_shadow_card(img, strip_box, radius=16)
    col_w = (strip_box[2] - strip_box[0]) / 3
    font_p_name = pil_font("semibold", 16)
    font_p_cond = pil_font("regular", 14)
    font_p_temp = pil_font("bold", 26)
    font_p_rain = pil_font("medium", 15)
    font_p_hours = pil_font("regular", 12)

    for i, p in enumerate(periods):
        cx0 = strip_box[0] + i * col_w
        cx1 = cx0 + col_w
        if i > 0:
            mid_x = cx0
            draw.line((mid_x, strip_box[1] + 16, mid_x, strip_box[3] - 16), fill=(213, 222, 231), width=1)
        cx = (cx0 + cx1) / 2
        cy = (strip_box[1] + strip_box[3]) / 2
        hours = f"{p['start']:d}am–{p['end'] - 12 if p['end'] > 12 else p['end']}{'am' if p['end'] <= 12 else 'pm'}"
        # prettier hours
        def fmt_hour(h: int, closed: bool = False) -> str:
            # end is exclusive; show as the clock hour itself (12pm, 6pm)
            display = h if not closed else h
            if display == 0 or display == 24:
                return "12am"
            if display == 12:
                return "12pm"
            if display < 12:
                return f"{display}am"
            return f"{display - 12}pm"

        hours = f"{fmt_hour(p['start'])}–{fmt_hour(p['end'])}"
        draw.text((cx, strip_box[1] + 10), p["name"].upper(), font=font_p_name, fill=NAVY_RGB, anchor="ma")
        draw.text((cx, strip_box[1] + 28), hours, font=font_p_hours, fill=MUTED_RGB, anchor="ma")

        kind = icon_kind(p["code"])
        icon = icons[kind].resize((52, 52), Image.Resampling.LANCZOS)
        img.paste(icon, (int(cx - 26), int(cy - 22)), icon)
        draw = ImageDraw.Draw(img)
        draw.text((cx, cy + 32), p["label"], font=font_p_cond, fill=(40, 60, 75), anchor="ma")
        draw.text((cx, strip_box[3] - 36), f"{p['temp_r']}°  ·  {p['pop_r']}%", font=font_p_temp, fill=INK_RGB, anchor="ma")

    # The last line used font_p_temp for the combined temp/rain which is a bit large; redraw footer only
    draw_footer(
        draw,
        ["Data: Open-Meteo", "High / low °C  ·  rain chance  ·  wind and gusts in mph  ·  periods are town averages"],
        footer_h,
    )
    # Footer may have overdrawn; the strip already ends above it.
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG", optimize=True)
    log(f"wrote {path} {img.size}")
    return path


# ---------------------------------------------------------------------------
# NASA GIBS satellite
# ---------------------------------------------------------------------------

def _image_is_blank(im: Image.Image) -> bool:
    """True if a GIBS tile is essentially no-data (black / uniform)."""
    rgb = im.convert("RGB")
    arr = np.asarray(rgb)
    # Near-black no-data pixels
    dark = np.all(arr < 12, axis=2)
    dark_frac = float(dark.mean())
    # Luminance variation — a real UK pass has cloud / land contrast
    lum = arr.astype(np.float32).mean(axis=2)
    std = float(lum.std())
    # Reject almost-empty or flat images. A valid cloudy UK scene is bright and varied.
    if dark_frac > 0.92:
        return True
    if std < 8:
        return True
    # If almost the whole frame is near-black and the rest is nearly black too
    if lum.mean() < 8:
        return True
    return False


def fetch_gibs_layer(layer_id: str, day: date) -> Image.Image | None:
    lat0, lat1 = GIBS_FETCH_LAT
    lon0, lon1 = GIBS_FETCH_LON
    w, h = GIBS_FETCH_SIZE
    # WMS 1.3.0 + EPSG:4326 uses lat,lon axis order in BBOX
    params = (
        f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer_id}&STYLES=&FORMAT=image/png&TRANSPARENT=false"
        f"&HEIGHT={h}&WIDTH={w}&CRS=EPSG:4326"
        f"&BBOX={lat0},{lon0},{lat1},{lon1}&TIME={day.isoformat()}"
    )
    url = f"{GIBS_WMS}?{params}"
    log(f"  GIBS {layer_id} {day.isoformat()}")
    try:
        resp = http_get(url, timeout=45, tries=3)
    except PipelineError as exc:
        log(f"    skipped: {exc}")
        return None
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ctype or "text" in ctype or resp.content[:50].lstrip().startswith(b"<"):
        log("    skipped: WMS exception / non-image response")
        return None
    try:
        from io import BytesIO

        im = Image.open(BytesIO(resp.content))
        im.load()
    except Exception as exc:
        log(f"    skipped: could not decode image ({exc})")
        return None
    if _image_is_blank(im):
        log("    skipped: blank / no-data image")
        return None
    return im.convert("RGB")


def find_latest_satellite(today: date, max_lookback: int = 5) -> tuple[Image.Image, date, str, str, bool]:
    """Return (image, date, layer_id, layer_label, fell_back).

    Tries today first, then yesterday, then further back (up to max_lookback
    days) so an unattended morning run still finds a usable pass.
    """
    for offset in range(0, max_lookback + 1):
        day = today - timedelta(days=offset)
        for layer_id, label in GIBS_LAYERS:
            im = fetch_gibs_layer(layer_id, day)
            if im is not None:
                return im, day, layer_id, label, offset > 0
    raise PipelineError(
        f"No usable NASA GIBS true-colour image of the UK for {today} "
        f"or the previous {max_lookback} days."
    )


def compose_satellite(
    raw: Image.Image,
    image_date: date,
    layer_label: str,
    *,
    requested: date,
    fell_back: bool,
    path: Path,
) -> Path:
    old_lat0, old_lat1 = GIBS_FETCH_LAT
    old_lon0, old_lon1 = GIBS_FETCH_LON
    lat0, lat1 = GIBS_CROP_LAT
    lon0, lon1 = GIBS_CROP_LON
    src_w, src_h = raw.size
    x0 = (lon0 - old_lon0) / (old_lon1 - old_lon0) * src_w
    x1 = (lon1 - old_lon0) / (old_lon1 - old_lon0) * src_w
    y0 = (old_lat1 - lat1) / (old_lat1 - old_lat0) * src_h
    y1 = (old_lat1 - lat0) / (old_lat1 - old_lat0) * src_h
    raw = raw.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))

    canvas = Image.new("RGB", (W, H), (14, 36, 51))
    top_bar, bot_bar = 145, 168
    avail_w, avail_h = W - 36, H - top_bar - bot_bar
    src_w, src_h = raw.size
    scale = min(avail_w / src_w, avail_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    sat = raw.resize((new_w, new_h), Image.Resampling.LANCZOS)
    ix = (W - new_w) // 2
    iy = top_bar + (avail_h - new_h) // 2
    canvas.paste(sat, (ix, iy))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((ix - 1, iy - 1, ix + new_w, iy + new_h), outline=(220, 230, 236))

    def lonlat_to_xy(lon, lat):
        px = ix + (lon - lon0) / (lon1 - lon0) * new_w
        py = iy + (lat1 - lat) / (lat1 - lat0) * new_h
        return px, py

    k_lons = (0.03, 1.45)
    k_lats = (50.91, 51.50)
    p1 = lonlat_to_xy(k_lons[0], k_lats[1])
    p2 = lonlat_to_xy(k_lons[1], k_lats[0])
    pad = 3
    xA, yA = p1[0] - pad, p1[1] - pad
    xB, yB = p2[0] + pad, p2[1] + pad
    L = 16
    col = (255, 220, 80)
    segs = [
        (xA, yA, xA + L, yA), (xA, yA, xA, yA + L),
        (xB, yA, xB - L, yA), (xB, yA, xB, yA + L),
        (xA, yB, xA + L, yB), (xA, yB, xA, yB - L),
        (xB, yB, xB - L, yB), (xB, yB, xB, yB - L),
    ]
    for seg in segs:
        draw.line(seg, fill=col, width=2)

    lab_font = pil_font("semibold", 18)
    label = "Kent"
    lx = (xA + xB) / 2
    ly = yA - 22
    tb = draw.textbbox((0, 0), label, font=lab_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx, ty = lx - tw / 2, ly - th
    if tx < ix + 4:
        tx = ix + 4
    if ty < iy + 4:
        ty = yA
        tx = xA - tw - 8
    draw.text((tx, ty), label, font=lab_font, fill=col, stroke_width=3, stroke_fill=(14, 36, 51))

    draw.text((56, 32), "Kent Weather", font=pil_font("bold", 40), fill=(255, 255, 255))
    draw.text((56, 86), "UK satellite image", font=pil_font("medium", 24), fill=(160, 220, 215))

    fy = H - bot_bar + 16
    draw.text((56, fy), format_long_date(image_date), font=pil_font("semibold", 24), fill=(255, 255, 255))
    if fell_back:
        note = (
            f"Latest available true-colour pass  ·  "
            f"{requested.strftime('%A')}'s image was not yet published"
        )
    else:
        note = "Latest available true-colour pass"
    draw.text((56, fy + 38), note, font=pil_font("regular", 16), fill=(180, 200, 210))
    draw.text(
        (56, fy + 68),
        f"Imagery: NASA Worldview/GIBS  ·  {layer_label}",
        font=pil_font("regular", 15),
        fill=(150, 175, 188),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG", optimize=True)
    log(f"wrote {path} {canvas.size}")
    return path


# ---------------------------------------------------------------------------
# NOAA GFS model chart (MSLP isobars + precipitation rate)
# ---------------------------------------------------------------------------

GFS_NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_NOMADS_PROD = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
GFS_AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFS_CREDIT = "Data: NOAA GFS"
# Download box (a bit wider than the drawn map so projected corners are filled)
GFS_FETCH = {"toplat": 66, "bottomlat": 40, "leftlon": -24, "rightlon": 20}
# Drawn map extent (lon0, lon1, lat0, lat1) in a Lambert conformal projection
GFS_EXTENT = (-10.5, 7.5, 48.4, 59.6)
GFS_CARTOPY_DIR = Path("/workspace/kent-weather/cartopy_data")
GFS_W, GFS_H = 1080, 1350
GFS_RAIN_LEVELS = [0.2, 0.5, 1, 2, 4, 8, 16, 32, 200]
GFS_RAIN_COLOURS = [
    "#7FC3EC",  # 0.2  light rain / drizzle
    "#3E9BE0",  # 0.5
    "#1F6FC5",  # 1
    "#1E9E57",  # 2
    "#9CCB2E",  # 4
    "#F6C82F",  # 8
    "#F2801E",  # 16
    "#D7263D",  # 32+
]
GFS_SNOW_COLOURS = [
    "#E7DDF6", "#CDB8EE", "#B094E3", "#9470D6", "#7B4FC6", "#6236AD", "#4B2390", "#371571",
]
GFS_MAP_CITIES = [
    # name, lon, lat, label dx, dy (points), ha
    ("London", -0.13, 51.51, -6, 7, "right"),
    ("Paris", 2.35, 48.86, 6, 5, "left"),
    ("Dublin", -6.26, 53.35, -6, 5, "right"),
    ("Edinburgh", -3.19, 55.95, 6, 5, "left"),
    ("Manchester", -2.24, 53.48, 6, 4, "left"),
    ("Amsterdam", 4.90, 52.37, 6, 4, "left"),
    ("Cardiff", -3.18, 51.48, -6, 5, "right"),
]
KENT_POINT = (0.85, 51.22)  # roughly the middle of Kent


def gfs_valid_time(day: date, hour_local: int = 12) -> datetime:
    """UTC valid time for `hour_local`:00 UK time on `day` (12:00 GMT / 12:00 BST)."""
    from datetime import timezone

    local = datetime(day.year, day.month, day.day, hour_local, 0, tzinfo=TZ)
    return local.astimezone(timezone.utc)


def gfs_run_candidates(now_utc: datetime, n: int = 6) -> list[datetime]:
    base = now_utc.replace(minute=0, second=0, microsecond=0)
    base = base.replace(hour=(base.hour // 6) * 6)
    return [base - timedelta(hours=6 * i) for i in range(n)]


def _gfs_names(run: datetime, fh: int) -> tuple[str, str, str]:
    ymd = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    return ymd, hh, f"gfs.t{hh}z.pgrb2.0p25.f{fh:03d}"


def _gfs_fetch_nomads(run: datetime, fh: int) -> bytes | None:
    ymd, hh, fname = _gfs_names(run, fh)
    idx = f"{GFS_NOMADS_PROD}/gfs.{ymd}/{hh}/atmos/{fname}.idx"
    try:
        r = requests.get(idx, timeout=20, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        log(f"    NOMADS idx check failed: {exc}")
        return None
    if r.status_code != 200:
        log(f"    NOMADS: {fname} ({ymd} {hh}Z) not available yet (HTTP {r.status_code})")
        return None
    params = {
        "dir": f"/gfs.{ymd}/{hh}/atmos",
        "file": fname,
        "var_PRMSL": "on", "var_PRATE": "on", "var_CSNOW": "on", "var_TMP": "on",
        "lev_mean_sea_level": "on", "lev_surface": "on",
        "lev_2_m_above_ground": "on", "lev_850_mb": "on",
        "subregion": "", **{k: str(v) for k, v in GFS_FETCH.items()},
    }
    for attempt in range(1, 4):
        try:
            resp = requests.get(GFS_NOMADS_FILTER, params=params, timeout=60,
                                headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200 and resp.content[:4] == b"GRIB":
                return resp.content
            log(f"    NOMADS filter HTTP {resp.status_code} (attempt {attempt})")
        except requests.RequestException as exc:
            log(f"    NOMADS filter failed (attempt {attempt}): {exc}")
        time.sleep(2 * attempt)
    return None


_GFS_AWS_WANTED = [
    ("PRMSL", "mean sea level", None),
    ("PRATE", "surface", "instant"),
    ("CSNOW", "surface", "instant"),
    ("TMP", "2 m above ground", None),
    ("TMP", "850 mb", None),
]


def _gfs_fetch_aws(run: datetime, fh: int) -> bytes | None:
    """Byte-range download of the needed GRIB messages from NOAA's open-data bucket."""
    ymd, hh, fname = _gfs_names(run, fh)
    base = f"{GFS_AWS}/gfs.{ymd}/{hh}/atmos/{fname}"
    try:
        r = requests.get(base + ".idx", timeout=20, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        log(f"    AWS idx failed: {exc}")
        return None
    if r.status_code != 200:
        log(f"    AWS: {fname} ({ymd} {hh}Z) not available (HTTP {r.status_code})")
        return None
    lines = [ln.split(":") for ln in r.text.strip().splitlines()]
    chunks = []
    for i, parts in enumerate(lines):
        if len(parts) < 6:
            continue
        var, lev, step = parts[3], parts[4], parts[5]
        for wvar, wlev, wstep in _GFS_AWS_WANTED:
            if var == wvar and lev == wlev and (wstep is None or "ave" not in step):
                start = int(parts[1])
                end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
                try:
                    resp = requests.get(base, timeout=60, headers={
                        "User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
                except requests.RequestException as exc:
                    log(f"    AWS range failed: {exc}")
                    return None
                if resp.status_code not in (200, 206):
                    return None
                chunks.append(resp.content)
                break
    return b"".join(chunks) if chunks else None


def _gfs_decode(blob: bytes) -> dict[str, np.ndarray]:
    import pygrib

    with tempfile.NamedTemporaryFile(suffix=".grb2") as tmp:
        tmp.write(blob)
        tmp.flush()
        grbs = pygrib.open(tmp.name)
        out: dict[str, np.ndarray] = {}
        lat0, lat1 = GFS_FETCH["bottomlat"], GFS_FETCH["toplat"]
        lon0, lon1 = GFS_FETCH["leftlon"], GFS_FETCH["rightlon"]
        for m in grbs:
            name = m.shortName
            step = getattr(m, "stepType", "instant")
            if step != "instant":
                continue
            if name == "prmsl":
                key = "mslp"
            elif name == "prate":
                key = "prate"
            elif name == "csnow":
                key = "csnow"
            elif name in ("2t", "t") and m.typeOfLevel == "heightAboveGround":
                key = "t2m"
            elif name == "t" and m.typeOfLevel == "isobaricInhPa" and int(m.level) == 850:
                key = "t850"
            else:
                continue
            if key in out:
                continue
            vals = m.values
            lats, lons = m.latlons()
            lons = np.where(lons > 180, lons - 360, lons)
            # Subset / reorder to the fetch box (AWS messages are global)
            lat_ok = (lats[:, 0] >= lat0) & (lats[:, 0] <= lat1)
            vals, lats, lons = vals[lat_ok], lats[lat_ok], lons[lat_ok]
            order = np.argsort(lons[0])
            vals, lats, lons = vals[:, order], lats[:, order], lons[:, order]
            lon_ok = (lons[0] >= lon0) & (lons[0] <= lon1)
            vals, lats, lons = vals[:, lon_ok], lats[:, lon_ok], lons[:, lon_ok]
            if lats[0, 0] > lats[-1, 0]:
                vals, lats, lons = vals[::-1], lats[::-1], lons[::-1]
            out[key] = np.ma.filled(np.asarray(vals, dtype=float), np.nan)
            out.setdefault("lats", lats[:, 0].copy())
            out.setdefault("lons", lons[0].copy())
        grbs.close()
    if "mslp" not in out or "prate" not in out:
        raise PipelineError("GFS GRIB is missing MSLP or precipitation rate")
    return out


def fetch_gfs(day: date, hour_local: int = 12, now_utc: datetime | None = None) -> dict[str, Any]:
    """Fetch the newest GFS run that covers `hour_local` on `day`.

    Tries the latest cycle first and steps back 6 h at a time (GFS appears
    on NOMADS roughly 3.5-5 h after the cycle time), NOMADS first then the
    NOAA open-data bucket on AWS.
    """
    from datetime import timezone

    return fetch_gfs_series(day, [hour_local], now_utc=now_utc)[0]


def _gfs_fetch_one(run: datetime, fh: int, valid: datetime, errors: list[str]) -> dict[str, Any] | None:
    for source, fetcher in (("NOMADS", _gfs_fetch_nomads), ("AWS", _gfs_fetch_aws)):
        log(f"  GFS {run:%Y-%m-%d %H}Z +{fh}h via {source}")
        blob = fetcher(run, fh)
        if not blob:
            errors.append(f"{run:%d %HZ}+{fh} {source}: unavailable")
            continue
        try:
            fields = _gfs_decode(blob)
        except Exception as exc:  # noqa: BLE001
            log(f"    decode failed: {exc}")
            errors.append(f"{run:%d %HZ}+{fh} {source}: decode failed ({exc})")
            continue
        fields.update({"run": run, "fh": fh, "valid": valid, "source": source})
        return fields
    return None


def fetch_gfs_series(day: date, hours_local: list[int], now_utc: datetime | None = None) -> list[dict[str, Any]]:
    """Fetch several valid times on `day`, all from the same (newest usable) GFS run.

    A run is used only if every requested hour is available from it (each
    hour tries NOMADS, then the AWS open-data bucket); otherwise the next
    older cycle is tried.
    """
    from datetime import timezone

    now_utc = now_utc or datetime.now(timezone.utc)
    valids = [gfs_valid_time(day, h) for h in hours_local]
    errors: list[str] = []
    for run in gfs_run_candidates(now_utc):
        fhs = [int((v - run).total_seconds() // 3600) for v in valids]
        if min(fhs) < 0:
            continue
        if any(fh > 384 or (fh > 120 and fh % 3) for fh in fhs):
            errors.append(f"{run:%d %HZ}: f{max(fhs)} out of range")
            continue
        out = []
        for fh, valid in zip(fhs, valids):
            fields = _gfs_fetch_one(run, fh, valid, errors)
            if fields is None:
                break
            out.append(fields)
        if len(out) == len(valids):
            return out
    raise PipelineError("No usable GFS run found: " + "; ".join(errors[-6:]))


def gfs_isobar_levels(mslp_hpa_fields: list[np.ndarray]) -> list[int]:
    lo = int(np.floor(min(np.nanmin(m) for m in mslp_hpa_fields) / 4) * 4)
    hi = int(np.ceil(max(np.nanmax(m) for m in mslp_hpa_fields) / 4) * 4)
    return list(range(lo, hi + 1, 4))


def _gfs_find_extrema(mslp_s: np.ndarray, lats: np.ndarray, lons: np.ndarray, extent) -> list[tuple[str, float, float, float]]:
    from scipy.ndimage import maximum_filter, minimum_filter

    size = 29  # ~7 degrees window
    lon0, lon1, lat0, lat1 = extent
    LON, LAT = np.meshgrid(lons, lats)
    inside = (LON > lon0 + 0.8) & (LON < lon1 - 0.8) & (LAT > lat0 + 0.6) & (LAT < lat1 - 0.6)
    out = []
    for kind, filt in (("H", maximum_filter), ("L", minimum_filter)):
        ext = filt(mslp_s, size=size, mode="nearest") == mslp_s
        ys, xs = np.where(ext & inside)
        picks = []
        for y, x in sorted(zip(ys, xs), key=lambda p: mslp_s[p] * (-1 if kind == "H" else 1)):
            lon, lat = LON[y, x], LAT[y, x]
            if any(abs(lon - a) < 4 and abs(lat - b) < 3 for _, a, b, _ in picks):
                continue
            # Skip flat "extrema" in a uniform field
            win = mslp_s[max(0, y - 14): y + 15, max(0, x - 14): x + 15]
            if (win.max() - win.min()) < 3:
                continue
            picks.append((kind, lon, lat, float(mslp_s[y, x])))
        out += picks[:3]
    return out


def render_gfs_map(fields: dict[str, Any], levels: list[int] | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Render the map panel (RGB PIL image) with cartopy."""
    import cartopy

    cartopy.config["data_dir"] = str(GFS_CARTOPY_DIR)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.ndimage import gaussian_filter

    lats, lons = fields["lats"], fields["lons"]
    mslp = fields["mslp"] / 100.0
    mslp_s = gaussian_filter(mslp, sigma=2.0, mode="nearest")
    rate = fields["prate"] * 3600.0  # kg m-2 s-1 -> mm/h
    rate = gaussian_filter(rate, sigma=0.8, mode="nearest")
    snow = fields.get("csnow")
    snow_mask = (snow > 0.5) if snow is not None else np.zeros_like(rate, dtype=bool)

    proj = ccrs.LambertConformal(central_longitude=-1.5, central_latitude=54.0,
                                 standard_parallels=(50, 58))
    pc = ccrs.PlateCarree()
    map_w, map_h = GFS_W, 950
    fig = plt.figure(figsize=(map_w / 100, map_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_extent(GFS_EXTENT, crs=pc)
    # Make the axes fill the panel exactly; extend the extent to the panel aspect
    x0, x1, y0, y1 = ax.get_extent(crs=proj)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    want = map_w / map_h
    w, h = x1 - x0, y1 - y0
    if w / h < want:
        w = h * want
    else:
        h = w / want
    ax.set_extent((cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), crs=proj)
    ax.set_facecolor(SEA)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m"),
                   facecolor=LAND, edgecolor="none", zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "50m"),
                   facecolor=SEA, edgecolor="none", zorder=1)

    cmap_r = ListedColormap(GFS_RAIN_COLOURS)
    cmap_s = ListedColormap(GFS_SNOW_COLOURS)
    norm = BoundaryNorm(GFS_RAIN_LEVELS, cmap_r.N)
    rain_only = np.where(~snow_mask, rate, np.nan)
    snow_only = np.where(snow_mask, rate, np.nan)
    ax.contourf(lons, lats, rain_only, levels=GFS_RAIN_LEVELS, cmap=cmap_r, norm=norm,
                transform=pc, zorder=2, alpha=0.88, extend="max")
    LON2, LAT2 = np.meshgrid(lons, lats)
    in_view = ((LON2 >= GFS_EXTENT[0]) & (LON2 <= GFS_EXTENT[1])
               & (LAT2 >= GFS_EXTENT[2]) & (LAT2 <= GFS_EXTENT[3]))
    has_snow = bool(((snow_mask & (rate >= GFS_RAIN_LEVELS[0]) & in_view).mean()) > 0.003)
    if has_snow:
        ax.contourf(lons, lats, snow_only, levels=GFS_RAIN_LEVELS, cmap=cmap_s, norm=norm,
                    transform=pc, zorder=2, alpha=0.9, extend="max")

    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
                   facecolor="none", edgecolor="#4A5A68", linewidth=0.9, zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m"),
                   facecolor="none", edgecolor="#8A97A3", linewidth=0.6, linestyle=(0, (3, 2)), zorder=3)

    # Isobars every 4 hPa, 1000 hPa slightly heavier
    if levels is None:
        levels = gfs_isobar_levels([mslp_s])
    widths = [1.9 if lv == 1000 else 1.25 for lv in levels]
    cs = ax.contour(lons, lats, mslp_s, levels=levels, colors=NAVY_DEEP, linewidths=widths,
                    transform=pc, zorder=4)
    # Several labels per isobar (clabel's default only labels each line once)
    import contourpy

    vx0, vx1, vy0, vy1 = ax.get_extent(crs=proj)
    mx, my = (vx1 - vx0) * 0.06, (vy1 - vy0) * 0.06
    spacing = (vx1 - vx0) * 0.38
    gen = contourpy.contour_generator(lons, lats, mslp_s)
    # Keep isobar labels away from town labels and the Kent tag (label sits to the side)
    place_boxes = []
    ux, uy = (vx1 - vx0), (vy1 - vy0)
    for _name, plon, plat, _dx, _dy, ha in GFS_MAP_CITIES:
        px, py = proj.transform_point(plon, plat, pc)
        side = -1 if ha == "right" else 1
        x_a, x_b = sorted((px - side * 0.03 * ux, px + side * 0.20 * ux))
        place_boxes.append((x_a, x_b, py - 0.035 * uy, py + 0.06 * uy))
    kx, ky = proj.transform_point(KENT_POINT[0], KENT_POINT[1], pc)
    place_boxes.append((kx - 0.06 * ux, kx + 0.17 * ux, ky - 0.07 * uy, ky + 0.05 * uy))

    def _clear_of_places(pt) -> bool:
        return not any(a0 <= pt[0] <= a1 and b0 <= pt[1] <= b1 for a0, a1, b0, b1 in place_boxes)

    manual = []
    for lv in levels:
        for seg in gen.lines(lv):
            if len(seg) < 4:
                continue
            pts = proj.transform_points(pc, seg[:, 0], seg[:, 1])[:, :2]
            ok = (pts[:, 0] > vx0 + mx) & (pts[:, 0] < vx1 - mx) & (pts[:, 1] > vy0 + my) & (pts[:, 1] < vy1 - my)
            vis = pts[ok]
            if len(vis) < 4:
                continue
            d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(vis, axis=0).T))])
            if d[-1] < spacing * 0.35:
                continue
            n = max(1, int(d[-1] // spacing))
            for k in range(n):
                target = d[-1] * (k + 0.5) / n
                j0 = min(int(np.searchsorted(d, target)), len(vis) - 1)
                # Walk outwards along the line to a spot clear of town names / Kent tag
                for step in range(0, len(vis)):
                    picked = None
                    for j in (j0 + step, j0 - step):
                        if 0 <= j < len(vis) and _clear_of_places(vis[j]):
                            picked = j
                            break
                    if picked is not None:
                        manual.append((float(vis[picked, 0]), float(vis[picked, 1])))
                        break
    try:
        labels = ax.clabel(cs, levels=levels, fmt="%d", inline=True, inline_spacing=6,
                           fontsize=12, manual=manual) if manual else []
    except Exception as exc:  # noqa: BLE001
        log(f"    manual isobar labels failed ({exc}); using defaults")
        labels = ax.clabel(cs, levels=levels, fmt="%d", inline=True, inline_spacing=6, fontsize=12)
    for t in labels:
        t.set_fontproperties(fp("semibold", 12.5))
        t.set_color(NAVY_DEEP)
        t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

    # Highs and lows
    extent_ll = GFS_EXTENT
    centres = _gfs_find_extrema(mslp_s, lats, lons, (extent_ll[0] - 2, extent_ll[1] + 2,
                                                     extent_ll[2] - 1, extent_ll[3] + 1))
    vx0, vx1, vy0, vy1 = ax.get_extent(crs=proj)
    mx, my = (vx1 - vx0) * 0.05, (vy1 - vy0) * 0.06
    visible = []
    for c in centres:
        px, py = proj.transform_point(c[1], c[2], pc)
        if vx0 + mx < px < vx1 - mx and vy0 + my < py < vy1 - my:
            visible.append(c)
    centres = visible
    for kind, lon, lat, val in centres:
        col = "#1F5FA8" if kind == "H" else "#C0392B"
        ax.text(lon, lat, kind, transform=pc, ha="center", va="center", zorder=7,
                fontproperties=fp("bold", 34), color=col, clip_on=True,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="white")])
        ax.text(lon, lat, f"\n\n{val:.0f}", transform=pc, ha="center", va="center", zorder=7,
                fontproperties=fp("bold", 14), color=col, clip_on=True,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])

    # Orientation cities
    for name, lon, lat, dx, dy, ha in GFS_MAP_CITIES:
        ax.plot(lon, lat, "o", ms=4.5, mfc="#3A4A58", mec="white", mew=1.0, transform=pc, zorder=8)
        ax.annotate(name, (lon, lat), xycoords=pc._as_mpl_transform(ax), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="bottom", zorder=8,
                    fontproperties=fp("medium", 12.5), color="#2E3E4C",
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    # Kent marker
    klon, klat = KENT_POINT
    ax.plot(klon, klat, marker="o", ms=17, mfc="none", mec="#F4B400", mew=3.2, transform=pc, zorder=9)
    ax.plot(klon, klat, marker="o", ms=6, mfc="#F4B400", mec=NAVY_DEEP, mew=1.0, transform=pc, zorder=9)
    ax.annotate("KENT", (klon, klat), xycoords=pc._as_mpl_transform(ax), xytext=(13, -5),
                textcoords="offset points", ha="left", va="top", zorder=9,
                fontproperties=fp("bold", 16), color=NAVY_DEEP,
                bbox=dict(boxstyle="round,pad=0.25", fc="#F4B400", ec=NAVY_DEEP, lw=1.2))

    ax.spines["geo"].set_visible(False)
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    panel = Image.open(buf).convert("RGB")
    if panel.size != (map_w, map_h):
        panel = panel.resize((map_w, map_h), Image.Resampling.LANCZOS)

    # Kent value summary for metadata
    iy = int(np.argmin(np.abs(lats - klat)))
    ix = int(np.argmin(np.abs(lons - klon)))
    meta = {
        "kent_mslp_hpa": round(float(mslp[iy, ix]), 1),
        "kent_precip_rate_mm_h": round(float(fields["prate"][iy, ix] * 3600.0), 2),
        "has_snow": has_snow,
        "centres": [
            {"type": k, "lon": round(float(a), 2), "lat": round(float(b), 2), "hpa": round(v)}
            for k, a, b, v in centres
        ],
    }
    return panel, meta


def _draw_rate_legend(draw: ImageDraw.ImageDraw, x: int, y: int, colours: list[str], title: str, show_ticks: bool) -> int:
    seg_w, seg_h = 92, 18
    draw.text((x, y), title, font=pil_font("semibold", 19), fill=NAVY_RGB)
    y += 32
    for i, col in enumerate(colours):
        x0 = x + i * seg_w
        rgb = tuple(int(col[j: j + 2], 16) for j in (1, 3, 5))
        draw.rectangle((x0, y, x0 + seg_w - 3, y + seg_h), fill=rgb)
    if show_ticks:
        tf = pil_font("medium", 15)
        ticks = ["0.2", "0.5", "1", "2", "4", "8", "16", "32+"]
        for i, t in enumerate(ticks):
            draw.text((x + i * seg_w, y + seg_h + 4), t, font=tf, fill=MUTED_RGB)
    return y + seg_h + (28 if show_ticks else 6)


def draw_gfs_chart(
    fields: dict[str, Any],
    day: date,
    path: Path,
    *,
    levels: list[int] | None = None,
    frame: tuple[int, int] | None = None,
    snow_legend: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Draw the GFS pressure & rain chart. With `frame=(i, n)` it draws an
    animation frame: big valid-time clock + progress dots in the header."""
    panel, meta = render_gfs_map(fields, levels=levels)
    img = Image.new("RGB", (GFS_W, GFS_H), SOFT_RGB)
    draw = ImageDraw.Draw(img)

    header_h = 198
    valid_local = fields["valid"].astimezone(TZ)
    zone = valid_local.tzname() or "UK"
    draw.rectangle((0, 0, GFS_W, header_h), fill=NAVY_RGB)
    draw.rectangle((0, header_h - 4, GFS_W, header_h), fill=TEAL_RGB)
    draw.text((56, 28), "Kent Weather", font=pil_font("bold", 54), fill=(255, 255, 255))
    draw.text((56, 96), "GFS model: pressure & rain", font=pil_font("medium", 28), fill=(200, 230, 228))
    run = fields["run"]
    if frame is None:
        draw.text(
            (56, 140),
            f"{format_long_date(day)}  ·  {valid_local:%H:%M} {zone}",
            font=pil_font("regular", 22), fill=(190, 210, 220),
        )
        # Badge: model run
        badge = f"{run:%HZ} run  ·  T+{fields['fh']}"
        bf = pil_font("semibold", 18)
        tb = draw.textbbox((0, 0), badge, font=bf)
        bw = tb[2] - tb[0] + 28
        bx1 = GFS_W - 48
        draw.rounded_rectangle((bx1 - bw, 40, bx1, 76), radius=18, fill=TEAL_RGB)
        draw.text((bx1 - bw + 14, 40 + (36 - (tb[3] - tb[1])) / 2 - tb[1]), badge, font=bf, fill=(255, 255, 255))
    else:
        i, n = frame
        draw.text((56, 140), f"{format_long_date(day)}  ·  hour by hour",
                  font=pil_font("regular", 22), fill=(190, 210, 220))
        # Big clock, right-aligned, fixed position so it never jumps
        bx1 = GFS_W - 48
        clock = f"{valid_local:%H:%M}"
        cf = pil_font("bold", 64)
        zf = pil_font("semibold", 22)
        zb = draw.textbbox((0, 0), zone, font=zf)
        cb = draw.textbbox((0, 0), "00:00", font=cf)  # fixed width reference
        zw = zb[2] - zb[0]
        cw = cb[2] - cb[0]
        box_x0 = bx1 - cw - zw - 22 - 10 - 22
        draw.rounded_rectangle((box_x0, 30, bx1, 122), radius=20, fill=TEAL_RGB)
        draw.text((box_x0 + 22 - cb[0], 76 - (cb[1] + cb[3]) / 2), clock, font=cf, fill=(255, 255, 255))
        draw.text((bx1 - 22 - zw - zb[0], 96 - zb[3]), zone, font=zf, fill=(205, 238, 235))
        # Progress dots
        r, gap = 7, 26
        total_w = (n - 1) * gap
        dx0 = (box_x0 + bx1) / 2 - total_w / 2
        for k in range(n):
            cx = dx0 + k * gap
            if k == i:
                draw.ellipse((cx - r, 150 - r, cx + r, 150 + r), fill=(255, 255, 255))
            else:
                draw.ellipse((cx - r + 2, 150 - r + 2, cx + r - 2, 150 + r - 2), outline=(150, 200, 200), width=2)

    map_y = header_h
    img.paste(panel, (0, map_y))
    draw.line((0, map_y + panel.size[1], GFS_W, map_y + panel.size[1]), fill=(200, 212, 222), width=2)

    y = map_y + panel.size[1] + 22
    if meta["has_snow"] if snow_legend is None else snow_legend:
        y = _draw_rate_legend(draw, 56, y, GFS_RAIN_COLOURS, "Rain (mm per hour)", True)
        y = _draw_rate_legend(draw, 56, y - 4, GFS_SNOW_COLOURS, "Snow", False)
    else:
        y = _draw_rate_legend(draw, 56, y, GFS_RAIN_COLOURS, "Rain and showers (mm per hour)", True)
    y += 8
    draw.text((56, y), "Lines: sea-level pressure (hPa)  ·  H = high, L = low  ·  closer lines = windier",
              font=pil_font("regular", 17), fill=INK_RGB)
    fy = GFS_H - 34
    run_txt = f"run {run:%d %b %H}Z UTC" + (f" +{fields['fh']}h" if frame is not None else "")
    draw.text((56, fy), f"{GFS_CREDIT}  ·  {run_txt}  ·  raw computer model output",
              font=pil_font("regular", 15), fill=MUTED_RGB)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG", optimize=True)
    log(f"wrote {path} {img.size}")
    meta.update({
        "model": "NOAA GFS 0.25°",
        "run_utc": run.strftime("%Y-%m-%dT%H:%MZ"),
        "forecast_hour": fields["fh"],
        "valid_utc": fields["valid"].strftime("%Y-%m-%dT%H:%MZ"),
        "valid_local": valid_local.isoformat(),
        "source": fields["source"],
        "credit": GFS_CREDIT,
    })
    return path, meta



GFS_LOOP_HOURS = [12, 13, 14, 15, 16]  # UK local time, hourly frames
GFS_LOOP_FRAME_S = 0.9
GFS_LOOP_LAST_S = 2.0
GFS_LOOP_MP4_PASSES = 2  # play the sequence twice in the MP4 (GIF loops forever)
GFS_LOOP_GIF_MAX = 8_000_000


def build_gfs_loop(day: date, out_dir: Path, *, gif: bool = True) -> tuple[list[Path], dict[str, Any]]:
    """Hourly GFS pressure & rain animation (12:00-16:00 UK time, one run)."""
    from scipy.ndimage import gaussian_filter

    t0 = time.time()
    log("Fetching NOAA GFS hourly frames for the loop…")
    series = fetch_gfs_series(day, GFS_LOOP_HOURS)
    levels = gfs_isobar_levels([gaussian_filter(f["mslp"] / 100.0, sigma=2.0, mode="nearest") for f in series])
    # Snow legend fixed across frames (shown if any frame has meaningful snow)
    snow_any = False
    n = len(series)
    out_files: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="kw_gfs_loop_") as tmp:
        tmpd = Path(tmp)
        frame_paths, metas = [], []
        for i, f in enumerate(series):
            pth, meta = draw_gfs_chart(f, day, tmpd / f"frame_{i:02d}.png", levels=levels, frame=(i, n),
                                       snow_legend=False)
            frame_paths.append(pth)
            metas.append(meta)
        if any(m["has_snow"] for m in metas):
            snow_any = True
            frame_paths, metas = [], []
            for i, f in enumerate(series):
                pth, meta = draw_gfs_chart(f, day, tmpd / f"frame_{i:02d}.png", levels=levels, frame=(i, n),
                                           snow_legend=True)
                frame_paths.append(pth)
                metas.append(meta)

        # --- MP4 (H.264 / yuv420p / 1080x1350) ---
        if not which("ffmpeg"):
            raise PipelineError("ffmpeg is not installed")
        fps = 10
        per = max(1, round(GFS_LOOP_FRAME_S * fps))
        last = max(per, round(GFS_LOOP_LAST_S * fps))
        seq: list[Path] = []
        for _ in range(GFS_LOOP_MP4_PASSES):
            for i, fpth in enumerate(frame_paths):
                seq += [fpth] * (last if i == n - 1 else per)
        seq_dir = tmpd / "seq"
        seq_dir.mkdir()
        for k, src in enumerate(seq):
            os.symlink(src, seq_dir / f"{k:04d}.png")
        mp4 = out_dir / "gfs_loop.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(seq_dir / "%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-crf", "20", "-preset", "slow", "-r", str(fps),
            "-movflags", "+faststart", "-an", str(mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise PipelineError(f"ffmpeg failed: {proc.stderr.strip() or proc.returncode}")
        out_files.append(mp4)
        log(f"wrote {mp4} ({mp4.stat().st_size} bytes)")

        # --- GIF (loops forever) ---
        gif_info: dict[str, Any] = {"made": False}
        if gif:
            try:
                ims = [Image.open(fp_).convert("RGB") for fp_ in frame_paths]
                # One shared adaptive palette so colours don't flicker between frames
                strip = Image.new("RGB", (ims[0].width, ims[0].height * len(ims)))
                for k, im in enumerate(ims):
                    strip.paste(im, (0, k * ims[0].height))
                pal = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
                durations = [int(GFS_LOOP_FRAME_S * 1000)] * (n - 1) + [int(GFS_LOOP_LAST_S * 1000)]
                gpath = out_dir / "gfs_loop.gif"
                q[0].save(gpath, save_all=True, append_images=q[1:], duration=durations, loop=0,
                          optimize=True, disposal=1)
                size = gpath.stat().st_size
                if size > GFS_LOOP_GIF_MAX:
                    log(f"  GIF is {size} bytes (> {GFS_LOOP_GIF_MAX}); dropping it")
                    gpath.unlink()
                    gif_info = {"made": False, "reason": f"too large ({size} bytes)"}
                else:
                    out_files.append(gpath)
                    gif_info = {"made": True, "bytes": size}
                    log(f"wrote {gpath} ({size} bytes)")
            except Exception as exc:  # noqa: BLE001
                log(f"  GIF failed: {exc}")
                gif_info = {"made": False, "reason": str(exc)}

    duration = None
    if which("ffprobe"):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
                            capture_output=True, text=True)
        try:
            duration = round(float(pr.stdout.strip()), 2)
        except ValueError:
            pass
    run = series[0]["run"]
    meta = {
        "run_utc": run.strftime("%Y-%m-%dT%H:%MZ"),
        "source": sorted({f["source"] for f in series}),
        "frames": [
            {"valid_local": f["valid"].astimezone(TZ).isoformat(), "forecast_hour": f["fh"]}
            for f in series
        ],
        "frame_seconds": GFS_LOOP_FRAME_S,
        "last_frame_seconds": GFS_LOOP_LAST_S,
        "mp4": {"bytes": mp4.stat().st_size, "duration_s": duration, "passes": GFS_LOOP_MP4_PASSES,
                "fps": fps, "size": [GFS_W, GFS_H]},
        "gif": gif_info,
        "isobar_levels_hpa": [levels[0], levels[-1]],
        "snow_legend": snow_any,
        "credit": GFS_CREDIT,
        "seconds_taken": round(time.time() - t0, 1),
    }
    return out_files, meta


# ---------------------------------------------------------------------------
# Animated rain / temperature map
# ---------------------------------------------------------------------------

def _iter_polys(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        try:
            for g in geom.geoms:
                yield from _iter_polys(g)
        except Exception:
            return


def _iter_lines(geom):
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield g
    else:
        try:
            for g in geom.geoms:
                yield from _iter_lines(g)
        except Exception:
            return


def load_map_layers(repo: Path):
    data = data_dir(repo)
    kent = shape(json.loads((data / "kent_ceremonial_osm.geojson").read_text())["features"][0]["geometry"])
    coast_fc = json.loads((data / "ne_coast_se.geojson").read_text())
    land_fc = json.loads((data / "ne_land_se.geojson").read_text())
    coasts = [shape(f["geometry"]) for f in coast_fc["features"]]
    lands = [shape(f["geometry"]) for f in land_fc["features"]]
    return kent, coasts, lands


def prepare_interp_grid(grid):
    nx, ny = 240, 160
    gx = np.linspace(MAP_LON0, MAP_LON1, nx)
    gy = np.linspace(MAP_LAT0, MAP_LAT1, ny)
    GX, GY = np.meshgrid(gx, gy)
    points = np.column_stack([grid["lons"], grid["lats"]])
    return GX, GY, points


def interpolate_field(points, values, GX, GY, method="linear"):
    z = griddata(points, values, (GX, GY), method=method)
    if np.isnan(z).any():
        z_n = griddata(points, values, (GX, GY), method="nearest")
        z = np.where(np.isnan(z), z_n, z)
    return z


def setup_weather_cmaps():
    temp_cmap = LinearSegmentedColormap.from_list(
        "kent_temp",
        [
            (0.00, "#3B2D6B"),
            (0.15, "#2C4B8C"),
            (0.30, "#2E7EA8"),
            (0.45, "#4EA3A0"),
            (0.58, "#8FBE7A"),
            (0.70, "#E0C35A"),
            (0.85, "#E08A3C"),
            (1.00, "#C4472A"),
        ],
    )
    rain_colors = [
        (0.00, (0.10, 0.85, 0.95, 0.00)),
        (0.06, (0.10, 0.80, 0.95, 0.45)),
        (0.18, (0.05, 0.55, 0.95, 0.62)),
        (0.36, (0.25, 0.25, 0.95, 0.72)),
        (0.55, (0.55, 0.10, 0.85, 0.80)),
        (0.75, (0.75, 0.05, 0.55, 0.88)),
        (1.00, (0.55, 0.00, 0.30, 0.92)),
    ]
    rain_cmap = LinearSegmentedColormap.from_list("kent_rain", rain_colors)
    return temp_cmap, rain_cmap


def nice_temp_limits(temps: np.ndarray) -> tuple[float, float, list[int]]:
    """Adaptive colour scale that still stays readable year-round."""
    finite = temps[np.isfinite(temps)]
    if finite.size == 0:
        raise PipelineError("Hourly temperature grid is empty")
    lo = float(np.floor(finite.min() - 1))
    hi = float(np.ceil(finite.max() + 1))
    # Keep a useful span so weak gradients don't look washed out
    if hi - lo < 12:
        mid = 0.5 * (hi + lo)
        lo, hi = mid - 6, mid + 6
        lo, hi = math.floor(lo), math.ceil(hi)
    span = hi - lo
    step = 4 if span > 14 else 3 if span > 10 else 2
    first = int(math.ceil(lo / step) * step)
    ticks = list(range(first, int(hi) + 1, step))
    # Include the scale ends only when they are not cramped against a neighbour
    if not ticks or ticks[0] - lo >= max(2, step * 0.6):
        ticks = [int(lo)] + ticks
    if ticks[-1] < hi - max(1.5, step * 0.6):
        ticks.append(int(hi))
    return lo, hi, ticks


def nice_rain_limit(rains: np.ndarray) -> tuple[float, list[float]]:
    finite = rains[np.isfinite(rains)]
    rmax = float(np.nanmax(finite)) if finite.size else 0.0
    vmax = max(2.5, math.ceil(rmax * 2) / 2.0)
    # Don't let a single extreme spike stretch the scale too far
    vmax = min(vmax, 8.0) if rmax <= 8 else vmax
    if vmax <= 2.5:
        ticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    elif vmax <= 4:
        ticks = [0, 1, 2, 3, 4]
    else:
        step = 1 if vmax <= 6 else 2
        ticks = [i * step for i in range(0, int(vmax / step) + 1)]
        if ticks[-1] < vmax:
            ticks.append(vmax)
    return vmax, ticks


def friendly_timestamp(iso: str) -> tuple[str, str]:
    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M")
    date_line = dt.strftime("%A %-d %B")
    hour = dt.strftime("%-I:%M%p").lower()
    return date_line, f"{hour}  ·  UK time"


def select_animation_frames(times: list[str], start: date, end: date) -> list[int]:
    """3-hourly frames from 00:00 on start through 21:00 on end, inclusive."""
    start_ts = f"{start.isoformat()}T00:00"
    end_ts = f"{end.isoformat()}T21:00"
    idxs = []
    for i, t in enumerate(times):
        dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
        if dt.hour % 3 != 0:
            continue
        if t < start_ts or t > end_ts:
            continue
        idxs.append(i)
    if len(idxs) < 8:
        raise PipelineError(
            f"Not enough hourly frames for the animation ({len(idxs)} in {start}–{end})"
        )
    return idxs


def render_animation(grid: dict, repo: Path, start: date, end: date, frame_dir: Path) -> tuple[list[Path], dict]:
    kent, coasts, lands = load_map_layers(repo)
    GX, GY, points = prepare_interp_grid(grid)
    temp_cmap, rain_cmap = setup_weather_cmaps()

    idxs = select_animation_frames(grid["times"], start, end)
    log(f"animation frames: {len(idxs)} from {grid['times'][idxs[0]]} to {grid['times'][idxs[-1]]}")

    # Colour scales from the frames we will actually show
    sample_temps = grid["temp"][:, idxs]
    sample_rains = np.clip(grid["precip"][:, idxs], 0, None)
    t_vmin, t_vmax, t_ticks = nice_temp_limits(sample_temps)
    r_vmax, r_ticks = nice_rain_limit(sample_rains)
    r_vmin = 0.0

    fields = []
    for i in idxs:
        tfield = interpolate_field(points, grid["temp"][:, i], GX, GY, "linear")
        rfield = interpolate_field(points, grid["precip"][:, i], GX, GY, "linear")
        rfield = np.clip(rfield, 0, None)
        fields.append((tfield, rfield, grid["times"][i]))

    dpi = 120
    fig_w = fig_h = 1080 / dpi
    aspect = 1.0 / math.cos(math.radians(51.2))

    for p in frame_dir.glob("*.png"):
        p.unlink()
    frame_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    peak_rain = 0.0
    peak_rain_at = None
    tmin_seen = float("inf")
    tmax_seen = float("-inf")

    for fi, (tfield, rfield, iso) in enumerate(fields):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=SOFT)
        ax = fig.add_axes([0.06, 0.225, 0.88, 0.635])
        ax.set_facecolor(SEA)
        ax.set_xlim(MAP_LON0, MAP_LON1)
        ax.set_ylim(MAP_LAT0, MAP_LAT1)
        ax.set_aspect(aspect)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#9BB0BE")
            spine.set_linewidth(0.8)

        mesh = ax.pcolormesh(
            GX, GY, tfield, cmap=temp_cmap, vmin=t_vmin, vmax=t_vmax, shading="auto", zorder=1
        )
        rain_masked = np.ma.masked_less(rfield, 0.05)
        ax.pcolormesh(
            GX, GY, rain_masked, cmap=rain_cmap, vmin=r_vmin, vmax=r_vmax, shading="auto", zorder=2
        )
        rmax = float(np.nanmax(rfield))
        if rmax >= 0.2:
            # Scale contour levels to the colour bar so heavy-rain weeks still read clearly
            base = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            levels = [lv for lv in base if lv < r_vmax]
            ax.contour(
                GX, GY, rfield, levels=levels,
                colors=["#3b0a5c"],
                linewidths=[0.5, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5][: len(levels)],
                alpha=0.45, zorder=3,
            )
        if rmax > peak_rain:
            peak_rain = rmax
            peak_rain_at = iso
        tmin_seen = min(tmin_seen, float(np.nanmin(tfield)))
        tmax_seen = max(tmax_seen, float(np.nanmax(tfield)))

        kent_buf = kent.buffer(0.025)
        for c in coasts:
            clipped = c.difference(kent_buf)
            for line in _iter_lines(clipped):
                x, y = line.xy
                ax.plot(x, y, color="#5A6B78", lw=0.8, zorder=4, solid_capstyle="round", alpha=0.85)
        for land in lands:
            clipped = land.boundary.difference(kent_buf)
            for line in _iter_lines(clipped):
                x, y = line.xy
                ax.plot(x, y, color="#5A6B78", lw=0.55, zorder=4, alpha=0.55)

        for poly in _iter_polys(kent):
            x, y = poly.exterior.xy
            ax.plot(x, y, color=NAVY, lw=2.2, zorder=5, solid_capstyle="round")
            for ring in poly.interiors:
                xi, yi = ring.xy
                ax.plot(xi, yi, color=NAVY, lw=1.4, zorder=5)

        for name, lon, lat, ha, va, dx, dy in TOWNS:
            ax.plot(
                lon, lat, "o", ms=4.5, color="white",
                markeredgecolor=NAVY, markeredgewidth=1.0, zorder=6,
            )
            txt = ax.text(
                lon + dx, lat + dy, name,
                fontproperties=fp("medium", 8.5), color=NAVY,
                ha=ha, va=va, zorder=7,
            )
            txt.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground="white")])

        fig.text(0.07, 0.945, "Kent Weather", fontproperties=fp("bold", 20), color=NAVY, ha="left", va="center")
        fig.text(0.07, 0.905, "Rain & temperature forecast", fontproperties=fp("medium", 12), color=TEAL, ha="left", va="center")
        date_line, time_line = friendly_timestamp(iso)
        fig.text(0.93, 0.945, date_line, fontproperties=fp("semibold", 13), color=NAVY, ha="right", va="center")
        fig.text(0.93, 0.905, time_line, fontproperties=fp("regular", 11), color=MUTED, ha="right", va="center")

        if rmax >= 0.1:
            status, status_col = f"Rain up to {rmax:.1f} mm/h", "#5B1478"
        else:
            status, status_col = "No rain at this hour", MUTED
        fig.text(0.50, 0.195, status, fontproperties=fp("semibold", 10), color=status_col, ha="center", va="center")

        cax_t = fig.add_axes([0.07, 0.128, 0.40, 0.022])
        cax_r = fig.add_axes([0.53, 0.128, 0.40, 0.022])
        cb_t = fig.colorbar(mesh, cax=cax_t, orientation="horizontal")
        rain_sm = mpl.cm.ScalarMappable(norm=Normalize(r_vmin, r_vmax), cmap=rain_cmap)
        rain_sm.set_array([])
        cb_r = fig.colorbar(rain_sm, cax=cax_r, orientation="horizontal")
        cb_t.set_label("Temperature (°C)", fontproperties=fp("medium", 8), color=INK, labelpad=2)
        cb_r.set_label("Rain (mm per hour)", fontproperties=fp("medium", 8), color=INK, labelpad=2)
        for cb, ticks in ((cb_t, t_ticks), (cb_r, r_ticks)):
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{v:g}" for v in ticks])
            cb.ax.tick_params(labelsize=8, length=2, colors=INK, pad=2)
            for lab in cb.ax.get_xticklabels():
                lab.set_fontproperties(fp("regular", 8))

        fig.text(0.07, 0.052, "Data: Open-Meteo", fontproperties=fp("semibold", 8.5), color=NAVY, ha="left", va="center")
        fig.text(
            0.07, 0.028,
            "Kent outline: OpenStreetMap  ·  Coastline: Natural Earth  ·  Times in Europe/London",
            fontproperties=fp("regular", 7.5), color=MUTED, ha="left", va="center",
        )
        fig.text(
            0.93, 0.040, "Forecast field over Kent & nearby areas",
            fontproperties=fp("regular", 7.5), color=MUTED, ha="right", va="center",
        )

        out = frame_dir / f"frame_{fi:03d}.png"
        fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(out)
        if fi % 8 == 0 or fi == len(fields) - 1:
            log(f"  frame {fi + 1}/{len(fields)} {iso}")

    meta = {
        "frame_count": len(saved),
        "first": grid["times"][idxs[0]],
        "last": grid["times"][idxs[-1]],
        "peak_rain_mm_h": round(peak_rain, 2),
        "peak_rain_at": peak_rain_at,
        "temp_min": round(tmin_seen, 1),
        "temp_max": round(tmax_seen, 1),
        "temp_scale": [t_vmin, t_vmax],
        "rain_scale_max": r_vmax,
    }
    return saved, meta


def encode_mp4(frame_paths: list[Path], dest: Path, fps: int = 6) -> dict:
    if not which("ffmpeg"):
        raise PipelineError("ffmpeg is not installed on this machine")
    # Hold first/last frames so the clip is easier to read (~10–12 s)
    hold_first, hold_last = 3, 4
    seq = [frame_paths[0]] * hold_first + frame_paths + [frame_paths[-1]] * hold_last
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kw_seq_") as tmp:
        seq_dir = Path(tmp)
        for i, src in enumerate(seq):
            os.symlink(src.resolve(), seq_dir / f"{i:04d}.png")
        # CRF 20 is a bit smaller than the original sample's 18 and still sharp
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(seq_dir / "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-preset", "slow",
            "-movflags", "+faststart",
            "-an",
            str(dest),
        ]
        log(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise PipelineError(f"ffmpeg failed: {proc.stderr.strip() or proc.returncode}")

    # Re-encode more aggressively if the file is unexpectedly large
    size = dest.stat().st_size
    if size > 5_500_000:
        log(f"  mp4 is {size} bytes; re-encoding at crf 23 to stay a few MB")
        tmp_out = dest.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(dest),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "slow",
            "-movflags", "+faststart", "-an",
            str(tmp_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            tmp_out.replace(dest)
            size = dest.stat().st_size
        else:
            tmp_out.unlink(missing_ok=True)

    # Probe duration
    duration = None
    if which("ffprobe"):
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(dest),
            ],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            try:
                duration = float(probe.stdout.strip())
            except ValueError:
                duration = None
    if duration and duration > 16:
        raise PipelineError(f"Animation is {duration:.1f}s; expected under ~15s")
    log(f"wrote {dest} ({size} bytes, {duration:.1f}s)" if duration else f"wrote {dest} ({size} bytes)")
    return {"bytes": size, "duration_s": duration, "fps": fps, "frames": len(seq)}


# ---------------------------------------------------------------------------
# Git publish + URL verification
# ---------------------------------------------------------------------------

def ensure_git_identity(repo: Path) -> None:
    def get(key: str) -> str:
        r = subprocess.run(["git", "config", key], cwd=repo, capture_output=True, text=True)
        return r.stdout.strip()

    if not get("user.name"):
        subprocess.run(["git", "config", "user.name", "Kent Weather"], cwd=repo, check=True)
    if not get("user.email"):
        subprocess.run(
            ["git", "config", "user.email", "rich0887-alt@users.noreply.github.com"],
            cwd=repo, check=True,
        )


def git_run(repo: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True)


def prune_old_posts(repo: Path, today: date, keep_days: int = 30) -> list[str]:
    posts = repo / "posts"
    if not posts.exists():
        return []
    cutoff = today - timedelta(days=keep_days)
    removed = []
    for child in sorted(posts.iterdir()):
        if not child.is_dir():
            continue
        try:
            d = date.fromisoformat(child.name)
        except ValueError:
            continue
        if d < cutoff:
            shutil.rmtree(child)
            removed.append(str(child.relative_to(repo)))
            log(f"pruned {child.relative_to(repo)} (older than {keep_days} days)")
    return removed


def publish_repo(repo: Path, paths: list[Path], message: str, prune: list[str]) -> str:
    ensure_git_identity(repo)
    log("git pull --rebase")
    pull = git_run(repo, ["pull", "--rebase", "origin", "main"], check=False)
    if pull.returncode != 0:
        raise PipelineError(f"git pull --rebase failed:\n{pull.stderr or pull.stdout}")
    rels = []
    for p in paths:
        rel = p.resolve().relative_to(repo.resolve())
        git_run(repo, ["add", "--", str(rel)])
        rels.append(str(rel))
    for rel in prune:
        git_run(repo, ["add", "-u", "--", rel])
    # Also pick up scripts / docs if they were edited as part of this working tree
    status = git_run(repo, ["status", "--porcelain"])
    if not status.stdout.strip():
        log("nothing new to commit")
        return git_run(repo, ["rev-parse", "HEAD"]).stdout.strip()
    git_run(repo, ["commit", "-m", message])
    sha = git_run(repo, ["rev-parse", "HEAD"]).stdout.strip()
    log(f"git push origin main ({sha[:8]})")
    push = git_run(repo, ["push", "origin", "main"], check=False)
    if push.returncode != 0:
        raise PipelineError(f"git push failed:\n{push.stderr or push.stdout}")
    return sha


def verify_raw_urls(urls: list[str], timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    pending = list(urls)
    attempt = 0
    while pending:
        attempt += 1
        still = []
        for url in pending:
            try:
                resp = requests.get(
                    url,
                    timeout=20,
                    headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
                )
                if resp.status_code == 200 and resp.content:
                    log(f"  verified {url} ({len(resp.content)} bytes)")
                    continue
                log(f"  {url} -> HTTP {resp.status_code}, retrying")
            except requests.RequestException as exc:
                log(f"  {url} -> {exc}, retrying")
            still.append(url)
        pending = still
        if not pending:
            return
        if time.time() >= deadline:
            raise PipelineError(
                "Raw GitHub URLs did not return HTTP 200 within "
                f"{timeout_s}s:\n  " + "\n  ".join(pending)
            )
        time.sleep(min(5, max(1, deadline - time.time())))


# ---------------------------------------------------------------------------
# File info + JSON summary
# ---------------------------------------------------------------------------

def file_entry(repo: Path, path: Path) -> dict[str, Any]:
    rel = str(path.resolve().relative_to(repo.resolve()))
    suffix = path.suffix.lower()
    if suffix == ".png":
        kind, mime = "image", "image/png"
    elif suffix == ".mp4":
        kind, mime = "video", "video/mp4"
    elif suffix == ".gif":
        kind, mime = "image", "image/gif"
    else:
        kind, mime = "file", "application/octet-stream"
    info: dict[str, Any] = {
        "name": path.name,
        "type": kind,
        "mime": mime,
        "path": rel,
        "url": public_url(rel),
        "bytes": path.stat().st_size,
    }
    if suffix in (".png", ".gif"):
        with Image.open(path) as im:
            info["width"], info["height"] = im.size
    elif suffix == ".mp4" and which("ffprobe"):
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name,pix_fmt",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            data = json.loads(probe.stdout)
            stream = (data.get("streams") or [{}])[0]
            fmt = data.get("format") or {}
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
            info["codec"] = stream.get("codec_name")
            info["pix_fmt"] = stream.get("pix_fmt")
            try:
                info["duration_s"] = round(float(fmt.get("duration")), 2)
            except (TypeError, ValueError):
                pass
    return info


def emit_summary(summary: dict) -> None:
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _daily_satellite(today: date, out_dir: Path) -> tuple[Path, dict]:
    log("Fetching NASA GIBS true-colour satellite image…")
    raw, sat_date, layer_id, layer_label, fell_back = find_latest_satellite(today)
    sat_path = out_dir / "satellite.png"
    compose_satellite(
        raw, sat_date, layer_label,
        requested=today, fell_back=fell_back, path=sat_path,
    )
    return sat_path, {
        "date": sat_date.isoformat(),
        "layer": layer_id,
        "layer_label": layer_label,
        "fell_back": fell_back,
        "credit": "Imagery: NASA Worldview/GIBS",
    }


def _daily_gfs(today: date, out_dir: Path) -> tuple[Path, dict]:
    log("Fetching NOAA GFS model data (MSLP + precipitation rate)…")
    fields = fetch_gfs(today, hour_local=12)
    return draw_gfs_chart(fields, today, out_dir / "gfs_chart.png")


def mode_daily(repo: Path, today: date, out_dir: Path, image: str = "gfs", gfs_loop: bool = False) -> tuple[list[Path], dict]:
    locs = fetch_towns(forecast_days=16)
    town_rows = []
    for name, loc in zip(TOWN_NAMES, locs):
        row = town_daily_row(loc, today)
        row["name"] = name
        town_rows.append(row)
        log(
            f"  {name}: {row['label']}  {row['tmax_r']}/{row['tmin_r']}°C  "
            f"rain {row['pop_r']}%  {row['precip']:.1f}mm  "
            f"wind {row['wind_r']} mph gusts {row['gusts_r']}"
        )
    hourlies = [hourly_by_date(loc) for loc in locs]
    period_rows = []
    for name, start_h, end_h in PERIODS:
        p = period_summary(hourlies, today, start_h, end_h)
        p["name"] = name
        p["start"] = start_h
        p["end"] = end_h
        period_rows.append(p)
        log(f"  {name}: {p['label']}  {p['temp_r']}°C  rain {p['pop_r']}%")

    chart = out_dir / "today_chart.png"
    draw_daily_chart(town_rows, period_rows, today, chart)

    # Second image: GFS model chart by default, satellite as fallback (or by flag)
    extra_files: list[Path] = []
    warnings: list[str] = []
    gfs_meta: dict | None = None
    sat_meta: dict | None = None
    order = {"gfs": ["gfs", "satellite"], "satellite": ["satellite"], "both": ["gfs", "satellite"],
             "none": []}[image]
    for kind in order:
        if kind == "satellite" and image == "gfs" and gfs_meta is not None:
            break  # GFS worked, no fallback needed
        try:
            if kind == "gfs":
                pth, gfs_meta = _daily_gfs(today, out_dir)
            else:
                pth, sat_meta = _daily_satellite(today, out_dir)
            extra_files.append(pth)
        except Exception as exc:  # noqa: BLE001 - never let the second image kill the post
            msg = f"{kind} image failed: {exc}"
            log(f"WARNING: {msg}")
            warnings.append(msg)
    if image == "gfs" and gfs_meta is None and sat_meta is not None:
        warnings.append("GFS chart unavailable; used the satellite image instead")
    if image != "none" and not extra_files:
        warnings.append("No second image; post today_chart.png only")

    # Optional extra: hourly GFS animation (never allowed to break the run)
    loop_meta: dict | None = None
    loop_files: list[Path] = []
    if gfs_loop:
        try:
            loop_files, loop_meta = build_gfs_loop(today, out_dir)
        except Exception as exc:  # noqa: BLE001
            msg = f"gfs loop failed: {exc}"
            log(f"WARNING: {msg}")
            warnings.append(msg)

    avg = average_daily(town_rows)
    forecast = {
        "date": today.isoformat(),
        "towns": [
            {
                "name": t["name"],
                "code": t["code"],
                "label": t["label"],
                "high_c": t["tmax_r"],
                "low_c": t["tmin_r"],
                "rain_chance_pct": t["pop_r"],
                "rain_mm": round(t["precip"], 1),
                "wind_mph": t["wind_r"],
                "gusts_mph": t["gusts_r"],
            }
            for t in town_rows
        ],
        "county_average": {
            "code": avg["code"],
            "label": avg["label"],
            "high_c": avg["tmax_r"],
            "low_c": avg["tmin_r"],
            "rain_chance_pct": avg["pop_r"],
            "rain_mm": round(avg["precip"], 1),
            "wind_mph": avg["wind_r"],
            "gusts_mph": avg["gusts_r"],
        },
        "periods": [
            {
                "name": p["name"],
                "hours": f"{p['start']:02d}:00-{p['end']:02d}:00",
                "code": p["code"],
                "label": p["label"],
                "temp_c": p["temp_r"],
                "temp_range_c": [p["tmin_r"], p["tmax_r"]],
                "rain_chance_pct": p["pop_r"],
            }
            for p in period_rows
        ],
        "second_image": (
            "gfs" if gfs_meta is not None else ("satellite" if sat_meta is not None else None)
        ),
        "gfs": gfs_meta,
        "gfs_loop": loop_meta,
        "satellite": sat_meta,
        "credits": [c for c in (
            "Data: Open-Meteo",
            GFS_CREDIT if (gfs_meta is not None or loop_meta is not None) else None,
            "Imagery: NASA Worldview/GIBS" if sat_meta is not None else None,
        ) if c],
        "warnings": warnings,
    }
    return [chart, *extra_files, *loop_files], forecast


def _days_from_towns(locs: list[dict], days: list[date]) -> list[dict]:
    out = []
    for day in days:
        rows = [town_daily_row(loc, day) for loc in locs]
        avg = average_daily(rows)
        out.append(avg)
        log(
            f"  {avg['date']}  {avg['label']}  {avg['tmax_r']}/{avg['tmin_r']}°C  "
            f"rain {avg['pop_r']}%  {avg['precip']:.1f}mm  codes={avg['codes']}"
        )
    return out


def mode_7day(repo: Path, today: date, start: date, out_dir: Path) -> tuple[list[Path], dict]:
    end = start + timedelta(days=6)
    if start < today:
        raise PipelineError("7-day start date is in the past")
    # Need enough forecast days from *today* to cover the window
    need = (end - today).days + 2
    locs = fetch_towns(forecast_days=max(16, need))
    days = _days_from_towns(locs, daterange(start, 7))

    chart = out_dir / "forecast_chart.png"
    draw_multiday_chart(
        days,
        title="7-day forecast",
        subtitle=format_range(start, end),
        footer_lines=[
            "Data: Open-Meteo",
            "Average of Ashford, Canterbury, Maidstone, Tunbridge Wells, Folkestone & Margate",
        ],
        path=chart,
    )

    grid = fetch_grid(forecast_days=max(16, need))
    with tempfile.TemporaryDirectory(prefix="kw_frames_") as tmp:
        frames, anim_meta = render_animation(grid, repo, start, end, Path(tmp))
        video = out_dir / "rain_temp_map.mp4"
        enc = encode_mp4(frames, video)

    forecast = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": [
            {
                "date": d["date"],
                "code": d["code"],
                "label": d["label"],
                "high_c": d["tmax_r"],
                "low_c": d["tmin_r"],
                "rain_chance_pct": d["pop_r"],
                "rain_mm": round(d["precip"], 1),
            }
            for d in days
        ],
        "animation": {**anim_meta, **enc},
    }
    return [chart, video], forecast


def mode_10day(repo: Path, today: date, out_dir: Path) -> tuple[list[Path], dict]:
    start = today
    end = start + timedelta(days=9)
    locs = fetch_towns(forecast_days=16)
    days = _days_from_towns(locs, daterange(start, 10))

    chart = out_dir / "forecast_chart.png"
    draw_multiday_chart(
        days,
        title="10-day forecast",
        subtitle=format_range(start, end),
        footer_lines=[
            "Data: Open-Meteo",
            "Average of Ashford, Canterbury, Maidstone, Tunbridge Wells, Folkestone & Margate",
            "Days 8–10 have lower confidence and may change",
        ],
        low_confidence_from=7,
        two_column=True,
        path=chart,
    )
    forecast = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "low_confidence_from": days[7]["date"],
        "days": [
            {
                "date": d["date"],
                "code": d["code"],
                "label": d["label"],
                "high_c": d["tmax_r"],
                "low_c": d["tmin_r"],
                "rain_chance_pct": d["pop_r"],
                "rain_mm": round(d["precip"], 1),
                "lower_confidence": i >= 7,
            }
            for i, d in enumerate(days)
        ],
    }
    return [chart], forecast


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Kent Weather Facebook graphics and publish them to GitHub."
    )
    p.add_argument("mode", choices=["daily", "7day", "10day"], help="Which graphic set to build")
    p.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        help="7-day window start date (default: tomorrow in Europe/London, i.e. the week ahead)",
    )
    p.add_argument(
        "--daily-image",
        choices=["gfs", "satellite", "both", "none"],
        default="gfs",
        help="Second image on the daily post (default: gfs model chart, falls back to satellite)",
    )
    p.add_argument(
        "--gfs-loop",
        action="store_true",
        help="Daily mode: also make gfs_loop.mp4 (+ gfs_loop.gif if < 8 MB), hourly GFS "
             "pressure & rain frames 12:00-16:00 UK time. Off by default; failure only warns.",
    )
    p.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Daily mode only: build the daily graphics for this date instead of today",
    )
    p.add_argument("--repo", help="Path to the kent-weather-images git repository")
    p.add_argument("--no-push", action="store_true", help="Generate files but do not commit or push")
    p.add_argument("--no-prune", action="store_true", help="Do not delete posts older than 30 days")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_fonts()
        repo = default_repo(args.repo)
        today = today_london()
        generated_at = datetime.now(TZ)

        start = None
        if args.mode == "7day":
            start = parse_iso_date(args.start) if args.start else today + timedelta(days=1)
        elif args.start:
            raise PipelineError("--start is only used with the 7day mode")
        if args.date:
            if args.mode != "daily":
                raise PipelineError("--date is only used with the daily mode")
            today = parse_iso_date(args.date)

        out_dir = repo / "posts" / today.isoformat() / args.mode
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"mode={args.mode}  today={today.isoformat()}  out={out_dir}")

        if args.mode == "daily":
            files, forecast = mode_daily(repo, today, out_dir, image=args.daily_image, gfs_loop=args.gfs_loop)
        elif args.mode == "7day":
            files, forecast = mode_7day(repo, today, start, out_dir)
        else:
            files, forecast = mode_10day(repo, today, out_dir)

        # Persist a raw machine copy of the forecast figures next to the graphics
        (out_dir / "forecast.json").write_text(json.dumps(forecast, indent=2) + "\n")

        pruned: list[str] = []
        sha = None
        if not args.no_push:
            if not args.no_prune:
                pruned = prune_old_posts(repo, today_london())
            if args.mode == "7day":
                msg = f"Add {today.isoformat()} 7-day graphics ({start.isoformat()}–{(start + timedelta(days=6)).isoformat()})"
            elif args.mode == "10day":
                msg = f"Add {today.isoformat()} 10-day graphics"
            else:
                msg = f"Add {today.isoformat()} daily graphics"
            sha = publish_repo(repo, files + [out_dir / "forecast.json"], msg, pruned)

        file_infos = [file_entry(repo, p) for p in files]
        if not args.no_push:
            verify_raw_urls([f["url"] for f in file_infos])

        summary = {
            "ok": True,
            "mode": args.mode,
            "generated_at": generated_at.isoformat(),
            "timezone": "Europe/London",
            "date": today.isoformat(),
            "folder": str(out_dir.relative_to(repo)),
            "repo": str(repo),
            "commit": sha,
            "pruned": pruned,
            "files": file_infos,
            "forecast": forecast,
            "warnings": forecast.get("warnings", []) if isinstance(forecast, dict) else [],
        }
        emit_summary(summary)
        return 0
    except PipelineError as exc:
        log(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        log(f"ERROR: unexpected failure: {exc}")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
