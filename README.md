# kent-weather-images

Public graphics for the Kent Weather Facebook page.

Generated files live under `posts/YYYY-MM-DD/<mode>/` and are served at:

`https://raw.githubusercontent.com/rich0887-alt/kent-weather-images/main/<path>`

The generator is `scripts/generate.py`. On this machine a stable wrapper lives at `/workspace/kent-weather/generate.py` (a symlink to the script in this repo). A virtualenv with the Python dependencies is at `/workspace/kent-weather/.venv`; the script will create and populate it if needed.

## Commands

```bash
VENV=/workspace/kent-weather/.venv/bin/python
SCRIPT=/workspace/kent-weather/generate.py

# Daily 8am post: today's town chart + latest NASA GIBS UK satellite image
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
| `daily` | `today_chart.png`, `satellite.png`, `forecast.json` |
| `7day` | `forecast_chart.png`, `rain_temp_map.mp4`, `forecast.json` |
| `10day` | `forecast_chart.png`, `forecast.json` |

Charts are 1080×1080 PNG. The 7-day animation is H.264 `yuv420p` with `+faststart`, about 10–12 seconds and around 1 MB.

Figures always come from live Open-Meteo data (six towns: Ashford, Canterbury, Maidstone, Tunbridge Wells, Folkestone, Margate) in the Europe/London timezone. Satellite imagery is NASA Worldview/GIBS true colour; the script tries today, then falls back to the most recent usable pass.

Posts older than 30 days are pruned automatically to keep the repository small.
