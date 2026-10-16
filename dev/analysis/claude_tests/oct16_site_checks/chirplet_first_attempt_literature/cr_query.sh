#!/bin/bash
# usage: cr_query.sh outdir rows "query1" "query2" ...
out=$1; rows=$2; shift 2; mkdir -p "$out"; i=0
for q in "$@"; do i=$((i+1)); f="$out/q$i.json"
  for attempt in 1 2 3 4; do
    curl -s --retry 3 --retry-delay 2 -G "https://api.crossref.org/works" --data-urlencode "query.bibliographic=$q" --data-urlencode "rows=$rows" -o "$f"
    [ -s "$f" ] && break
  done
  echo "## $q"
  if [ -s "$f" ]; then /home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python -I "$(dirname "$0")/parse_crossref.py" "$f"; else echo "EMPTY RESPONSE"; fi
done
