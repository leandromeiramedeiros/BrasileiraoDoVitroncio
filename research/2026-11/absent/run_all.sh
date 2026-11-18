#!/bin/bash
# Rebuild everything in /home/user/bdv-lab/absent (about 2 minutes, 1 process at a time).
set -e; cd "$(dirname "$0")"; PY=/home/user/bdv-lab/venv/bin/python
node build.mjs                                   # features_absent.csv, future_absent.csv, suspension-rule validation
$PY fit.py dev > work/dev_variants.log 2>&1      # DEV table (never reads HOLDOUT)
$PY fit.py ref 60                                # referee test (DEV)
$PY fit.py final susp_w,absP_w                   # preds_absent.csv, preds_absent_lineup.csv, coef_absent.json, future shifts
# (final scoring, done once: shared/score.py preds_absent.csv ; compare.py preds_absent.csv ../context/preds_ctx_core.csv
#  preds_absent_lineup.csv ../players/js/preds_lineup_js.csv) -> work/score_*_holdout_consultation.txt
