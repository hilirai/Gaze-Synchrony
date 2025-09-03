#!/bin/bash
# Simple wrapper script for researchers
# Place this script in the same directory as your videos/ and results/ folders

set -e  # Exit on any error

echo "DEBUG: Script started with arguments: $@"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_usage() {
    echo -e "${BLUE}Gaze Synchronization Analysis${NC}"
    echo "Usage: $0 <player1_video> <player2_video> <cooperation|competition>"
    echo ""
    echo "Example: $0 recording_player1.mp4 recording_player2.mp4 cooperation"
    echo "         $0 recording_player1.mp4 recording_player2.mp4 competition"
    echo ""
    echo "Requirements:"
    echo "- Make sure your videos are in the 'videos/' folder"
    echo "- Results will be saved to the 'results/' folder"
}

echo "DEBUG: Checking argument count: $#"
if [ $# -ne 3 ]; then
    echo "DEBUG: Wrong number of arguments"
    print_usage
    exit 1
fi
echo "DEBUG: Arguments validated"

# Validate analysis type
ANALYSIS_TYPE="$3"
if [[ "$ANALYSIS_TYPE" != "cooperation" && "$ANALYSIS_TYPE" != "competition" ]]; then
    echo "Error: Analysis type must be either 'cooperation' or 'competition'"
    print_usage
    exit 1
fi

# Get current directory (where videos/ and results/ should be)
PROJECT_DIR="$(pwd)"
VIDEOS_DIR="$PROJECT_DIR/videos"
# Check if this is a solo run (both videos contain "solo")
if [[ "$1" == *"solo"* && "$2" == *"solo"* ]]; then
    RESULTS_DIR="$PROJECT_DIR/results/solo_$ANALYSIS_TYPE"
else
    RESULTS_DIR="$PROJECT_DIR/results/$ANALYSIS_TYPE"
fi

# Check if directories exist
if [ ! -d "$VIDEOS_DIR" ]; then
    echo -e "${YELLOW}Creating videos/ directory...${NC}"
    mkdir -p "$VIDEOS_DIR"
    echo "Please place your video files in: $VIDEOS_DIR"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${YELLOW}Creating results/ directory...${NC}"
    mkdir -p "$RESULTS_DIR"
fi

# Check if videos exist
if [ ! -f "$VIDEOS_DIR/$1" ]; then
    echo "Error: Video '$1' not found in $VIDEOS_DIR"
    exit 1
fi

if [ ! -f "$VIDEOS_DIR/$2" ]; then
    echo "Error: Video '$2' not found in $VIDEOS_DIR"
    exit 1
fi

echo -e "${GREEN}Starting gaze analysis...${NC}"
echo "Videos: $1, $2"
echo "Analysis type: $ANALYSIS_TYPE"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Get the script directory to find run_gaze_analysis.sh
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RUN_GAZE_SCRIPT="$SCRIPT_DIR/scripts/run_gaze_analysis.sh"

# Check if run_gaze_analysis.sh exists
if [ ! -f "$RUN_GAZE_SCRIPT" ]; then
    echo "Error: run_gaze_analysis.sh not found at $RUN_GAZE_SCRIPT"
    exit 1
fi

# Set environment variables for the analysis script
export INPUT_DIR="$VIDEOS_DIR"
export OUTPUT_DIR="$RESULTS_DIR"
export WORK_DIR="$PROJECT_DIR/workspace"

echo -e "${BLUE}Running gaze analysis script...${NC}"
echo "DEBUG: INPUT_DIR=$INPUT_DIR"
echo "DEBUG: OUTPUT_DIR=$OUTPUT_DIR"
echo "DEBUG: WORK_DIR=$WORK_DIR"
echo "DEBUG: Video files: $1, $2"
echo "DEBUG: Analysis type: $ANALYSIS_TYPE"

# Run the gaze analysis script directly
"$RUN_GAZE_SCRIPT" "$1" "$2" "$ANALYSIS_TYPE"

echo -e "${GREEN}Analysis complete! Check the results/ folder for outputs.${NC}"