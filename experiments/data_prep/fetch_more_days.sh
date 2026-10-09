#!/usr/bin/env bash
# Fetch additional Google 2011 task_usage parts for more trace days, with a
# disk-space guard so the week-long runner never fills the root filesystem.
#   fetch_more_days.sh <first_part> <last_part>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/data/gct_raw"
mkdir -p "$DIR"
first="${1:-85}"; last="${2:-135}"
MIN_FREE_KB=3000000   # keep at least ~3 GB free
for i in $(seq "$first" "$last"); do
  p=$(printf "part-%05d.csv.gz" "$i")
  [ -s "$DIR/$p" ] && continue
  free=$(df -Pk "$DIR" | awk 'NR==2{print $4}')
  if [ "$free" -lt "$MIN_FREE_KB" ]; then
    echo "disk guard: only ${free}KB free, stopping fetch at part $i"; exit 0
  fi
  url="https://storage.googleapis.com/clusterdata-2011-2/task_usage/${p%.csv.gz}-of-00500.csv.gz"
  curl -sS -o "$DIR/$p" "$url" || { echo "fetch failed $p"; rm -f "$DIR/$p"; }
done
echo "fetch_more_days done ($first..$last); parts now $(ls "$DIR"/part-*.csv.gz | wc -l)"
