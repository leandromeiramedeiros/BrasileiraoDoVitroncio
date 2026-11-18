#!/usr/bin/env bash
# Fetch Wikipedia season pages (wikitext) via the MediaWiki API. <= 1 request/second, generic User-Agent.
#   bash code/fetch.sh [first_year] [last_year] [langs] [divs]   (default 2009 2026 "en pt" "A B")
# Output: raw/{en,pt}_{A,B}_YYYY.json   (API JSON; 'error' key if the page does not exist)
set -u
cd "$(dirname "$0")/.."
UA="bdv-research/1.0 (football research)"
Y0=${1:-2009}; Y1=${2:-2026}; LANGS=${3:-en pt}; DIVS=${4:-A B}
mkdir -p raw
get() {  # lang title outfile
  local lang=$1 title=$2 out=$3
  if [ -s "$out" ] && ! grep -qi "too many requests" "$out" && [ "${FORCE:-0}" != 1 ]; then return; fi
  for try in 1 2 3 4; do
    curl -sS -G -A "$UA" "https://$lang.wikipedia.org/w/api.php" \
      --data-urlencode "action=parse" --data-urlencode "page=$title" --data-urlencode "prop=wikitext" \
      --data-urlencode "redirects=1" --data-urlencode "format=json" --data-urlencode "formatversion=2" \
      --data-urlencode "maxlag=5" -o "$out"
    sleep "${SLEEP:-2}"
    if grep -qi "too many requests" "$out"; then echo "rate-limited ($lang $title), backing off"; sleep $((15*try)); continue; fi
    break
  done
  echo "$(date +%T) $lang $title -> $(wc -c <"$out") bytes"
}
for y in $(seq "$Y0" "$Y1"); do for d in $DIVS; do for l in $LANGS; do
  if [ "$l" = en ]; then get en "${y} Campeonato Brasileiro Série ${d}" "raw/en_${d}_${y}.json"
  else get pt "Campeonato Brasileiro de Futebol de ${y} - Série ${d}" "raw/pt_${d}_${y}.json"; fi
done; done; done
