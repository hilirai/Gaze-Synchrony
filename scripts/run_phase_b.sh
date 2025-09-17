#!/bin/bash
# Run Phase B only for a specific player

set -e

PLAYER="$1"  
VIDEO="$2" 
WORK_DIR="${WORK_DIR:-./workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

if [ -z "$PLAYER" ]; then
    echo "Usage: $0 <player_number> [video_file]"
    exit 1
fi

if [ ! -d "$WORK_DIR/player${PLAYER}/output" ]; then
    echo "Error: Workspace output for player $PLAYER not found. Run Phase A first."
    exit 1
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export VIDEO_DIR="$WORK_DIR/player${PLAYER}/video_parts"
export OUTPUT_DIR="$WORK_DIR/player${PLAYER}/output"

echo "=================================================="
echo "🌐  Running Phase B only for Player $PLAYER"
echo "=================================================="

python "$SCRIPT_DIR/analyze_player_gaze.py" --phase-b-only

echo "✅ Phase B completed. Results in $WORK_DIR/player${PLAYER}/output/gaze_data"
