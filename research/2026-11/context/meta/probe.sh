#!/bin/bash
# probe candidate slugs: prints slug, http code, number of events in 2024, league name
slug="$1"; yr="${2:-2024}"
out=$(curl -sS --max-time 90 -w '\n%{http_code}' "https://site.web.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard?dates=${yr}&limit=1000")
code=$(echo "$out" | tail -n1)
body=$(echo "$out" | sed '$d')
n=$(echo "$body" | /home/user/bdv-lab/venv/bin/python -I -c "
import sys,json
try:
    d=json.load(sys.stdin); print(len(d.get('events',[])), '|', (d.get('leagues') or [{}])[0].get('name'))
except Exception as e: print('ERR')
")
echo "$slug $yr $code $n"
