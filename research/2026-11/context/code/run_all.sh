#!/bin/bash
# Full rebuild of the context track (≈ 10 min, single process; downloads use at most 3 connections).
set -e
cd "$(dirname "$0")/.."
PY=/home/user/bdv-lab/venv/bin/python
./code/download.sh                       # raw/<slug>_<year>.json (only missing + current/previous year; --all to refresh everything)
$PY -I code/parse_all.py                 # all_matches.json, all_teams.json
$PY -W ignore code/ekf.py                # features_ratings.csv   (all-competitions EKF vs league-only EKF)
$PY -W ignore code/build_features.py     # features_context.csv   (schedule, geography, stakes)
$PY -W ignore code/run_final.py --holdout  # preds_*.csv, variants/, future/, coef/ + official scorer
$PY -W ignore code/make_core.py          # features_core.csv
$PY -W ignore code/report_tables.py      # tables used in REPORT.md
$PY -W ignore code/production_coefs.py   # coef/production_ctx_core.json
