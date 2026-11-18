#!/bin/bash
# Downloads yearly ESPN scoreboards for every non-league competition used by the context track.
#   ./code/download.sh            -> only missing files + current and previous year
#   ./code/download.sh --all      -> everything again
# At most 3 concurrent requests (protocol rule).
DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$DIR/raw"; mkdir -p "$RAW"
SLUGS="bra.copa_do_brazil conmebol.libertadores conmebol.sudamericana bra.camp.paulista bra.camp.carioca bra.camp.gaucho bra.camp.mineiro bra.copa_do_nordeste conmebol.recopa bra.supercopa_do_brazil fifa.cwc fifa.intercontinental_cup bra.3 bra.4"
THIS=$(date -u +%Y); ALL=0; [ "$1" = "--all" ] && ALL=1
fetch_one() {
  slug="$1"; y="$2"; dest="$3"
  for i in 1 2 3 4; do
    if curl -sSL --fail --max-time 240 -o "$dest.tmp" "https://site.web.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard?dates=${y}&limit=1000"; then
      mv "$dest.tmp" "$dest"; echo "ok $slug $y $(stat -c %s "$dest")"; return 0
    fi
    sleep $((2 ** i))
  done
  rm -f "$dest.tmp"; echo "FAIL $slug $y"; return 0
}
export -f fetch_one
jobs=""
for s in $SLUGS; do
  for y in $(seq 2006 "$THIS"); do
    dest="$RAW/${s}_${y}.json"
    if [ $ALL -eq 0 ] && [ "$y" -lt $((THIS - 1)) ] && [ -s "$dest" ]; then continue; fi
    jobs="$jobs$s $y $dest\n"
  done
done
printf "$jobs" | grep -c . 
printf "$jobs" | xargs -P 3 -L 1 bash -c 'fetch_one "$0" "$1" "$2"'
