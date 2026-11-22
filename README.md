# kent-weather-images

Public graphics for the Kent Weather Facebook page.

Generated files live under `posts/YYYY-MM-DD/<mode>/` and are served at:

`https://raw.githubusercontent.com/rich0887-alt/kent-weather-images/main/<path>`

The generator is `scripts/generate.py`. On this machine a stable wrapper lives at `/workspace/kent-weather/generate.py` (a symlink to the script in this repo). A virtualenv with the Python dependencies is at `/workspace/kent-weather/.venv`; the script will create and populate it if needed.

## Commands

```bash
VENV=/workspace/kent-weather/.venv/bin/python
SCRIPT=/workspace/kent-weather/generate.py

# Daily 8am post: today's town chart + NOAA GFS model chart (pressure & rain, 12:00 UK time)
# (--daily-image satellite|both|none to change; GFS failure falls back to the NASA satellite image)
$VENV $SCRIPT daily

# Sunday week-ahead post: 7-day chart + animated rain/temperature map
# Default start date is tomorrow (Europe/London)
$VENV $SCRIPT 7day
$VENV $SCRIPT 7day --start 2026-11-23

# Thursday outlook: 10-day chart starting today (days 8–10 marked lower confidence)
$VENV $SCRIPT 10day
```

Useful flags:

- `--start YYYY-MM-DD` — 7-day window start only (default: tomorrow in Europe/London)
- `--no-push` — write files locally but do not commit or push
- `--no-prune` — keep `posts/` folders older than 30 days
- `--repo PATH` — git repo root (default: this repository)

Progress and errors go to stderr. On success the script prints a JSON summary to stdout, including the public raw URLs and the exact Open-Meteo figures used in the graphics (so a caption can match). On failure it exits non-zero with a clear message.

Unless `--no-push` is set, each run does `git pull --rebase`, commits the new files, pushes to `main`, then waits up to 60 seconds for every raw URL to return HTTP 200.

## Output files

| Mode | Files |
| --- | --- |
| `daily` | `today_chart.png`, `gfs_chart.png` (or `satellite.png` as fallback), `forecast.json` |
| `7day` | `forecast_chart.png`, `rain_temp_map.mp4`, `forecast.json` |
| `10day` | `forecast_chart.png`, `forecast.json` |

Charts are 1080×1080 PNG. The 7-day animation is H.264 `yuv420p` with `+faststart`, about 10–12 seconds and around 1 MB.

Figures always come from live Open-Meteo data (six towns: Ashford, Canterbury, Maidstone, Tunbridge Wells, Folkestone, Margate) in the Europe/London timezone. The daily model chart uses NOAA GFS 0.25° data (MSLP + instantaneous precipitation rate, newest available run, from NOMADS with the NOAA AWS open-data bucket as backup; credit "Data: NOAA GFS"). Satellite imagery (fallback) is NASA Worldview/GIBS true colour; the script tries today, then falls back to the most recent usable pass.

Posts older than 30 days are pruned automatically to keep the repository small.

## GFS animation windows (`gfsloop` mode)

```bash
$VENV $SCRIPT gfsloop --start 08 --end 12   # Morning 08:00–12:00 (5 frames)
$VENV $SCRIPT gfsloop --start 12 --end 16   # Afternoon 12:00–16:00 (5 frames)
$VENV $SCRIPT gfsloop --start 16 --end 22   # Evening 16:00–22:00 (7 frames)
# optional: --date YYYY-MM-DD (default today, Europe/London); hours are UK local time (GMT/BST aware)
```

Output: `posts/<date>/gfsloop-HH00-HH00/gfs_loop.mp4` (+ `gfs_loop.gif` if < 8 MB) and `forecast.json`.
All frames come from one GFS run (newest run that has every frame; NOMADS then AWS).

## Git publishing

Every run takes an exclusive lock (`.git/kw-generate.lock`), commits its own outputs plus any
leftover generated files under `posts/`, then `git pull --rebase --autostash` and pushes, retrying
up to 3 times (with a stash fallback and automatic abort of half-finished rebases).
