#!/usr/bin/env bash
# ===========================================================================
#  Leave this running. It searches, evolves and reports on its own.
#  If Python crashes it restarts after 30s and resumes from the registry —
#  nothing already tried is ever tried again.
#
#      ./run_forever.sh              foreground
#      nohup ./run_forever.sh &      background, survives logout
#
#  Ctrl+C stops it cleanly.
# ===========================================================================
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}

trap 'echo; echo "stopping."; exit 0' INT TERM

while true; do
    echo "[$(date '+%F %T')] starting search..."
    "$PY" run_search.py --config config.yaml
    code=$?
    echo "[$(date '+%F %T')] exited with code $code"
    [ $code -eq 0 ] && { echo "clean exit, stopping."; break; }
    echo "restarting in 30s (Ctrl+C to stop)"
    sleep 30
done

echo "Leaderboard: $(pwd)/results/leaderboard.html"
