#!/bin/bash
# Gaze Synchronization Analysis Pipeline
# Runs inside Docker container with mounted volumes
#
# Expected Docker run command:
# docker run -it -v /host/videos:/app/input -v /host/results:/app/output -p 5000:5000 your-image
#
# Usage inside container: ./run_gaze_analysis.sh player1_video.mp4 player2_video.mp4

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories (set by start_analysis.sh)
INPUT_DIR="${INPUT_DIR:-./input}"       # Videos folder path
OUTPUT_DIR="${OUTPUT_DIR:-./output}"    # Results folder path  
WORK_DIR="${WORK_DIR:-./workspace}" # Temporary processing

print_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "🎥  GAZE SYNCHRONIZATION ANALYSIS PIPELINE  👁️"
    echo "=================================================="
    echo -e "${NC}"
}

log_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check if input directory exists and is mounted
    if [ ! -d "$INPUT_DIR" ]; then
        log_error "Input directory $INPUT_DIR not found. Make sure to mount host videos folder to /app/input"
        exit 1
    fi
    
    # Check if output directory exists and is mounted
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_error "Output directory $OUTPUT_DIR not found. Make sure to mount host results folder to /app/output"
        exit 1
    fi
    
    # Check if videos exist
    if [ ! -f "$INPUT_DIR/$1" ]; then
        log_error "Player 1 video '$1' not found in $INPUT_DIR"
        exit 1
    fi
    
    if [ ! -f "$INPUT_DIR/$2" ]; then
        log_error "Player 2 video '$2' not found in $INPUT_DIR"
        exit 1
    fi
    
    # Check if required scripts exist
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    if [ ! -f "$SCRIPT_DIR/analyze_player_gaze.py" ]; then
        log_error "Gaze tracking script not found. Looking for $SCRIPT_DIR/analyze_player_gaze.py"
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_DIR/sync_players_data.py" ]; then
        log_error "Sync script not found. Looking for $SCRIPT_DIR/sync_players_data.py"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

setup_workspace() {
    log_step "Setting up workspace..."
    
    # Create temporary workspace
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR/player1/video_parts"
    mkdir -p "$WORK_DIR/player2/video_parts" 
    mkdir -p "$WORK_DIR/player1/output"
    mkdir -p "$WORK_DIR/player2/output"
    
    # Create final output directories  
    mkdir -p "$OUTPUT_DIR/sync_results"
    mkdir -p "$OUTPUT_DIR/pair_01"  # For multi-condition analysis
    
    log_success "Workspace ready"
}

split_video() {
    local player=$1
    local video_file=$2
    local output_dir="$WORK_DIR/player${player}/video_parts"
    
    log_step "Splitting Player $player video into parts..."
    
    # Split video into 15-second segments (adjust as needed)
    ffmpeg -i "$INPUT_DIR/$video_file" \
           -c copy \
           -f segment \
           -segment_time 15 \
           -reset_timestamps 1 \
           "$output_dir/video_part_%03d.mp4" \
           -y -loglevel error
    
    local part_count=$(ls "$output_dir"/video_part_*.mp4 2>/dev/null | wc -l)
    log_success "Split Player $player video into $part_count parts"
}

run_gaze_tracking() {
    local player=$1
    local video_name=$2
    local analysis_type=$3
    
    log_step "Running gaze tracking for Player $player..."
    log_info "Video: $video_name"
    
    # Set up paths for the tracking script
    export VIDEO_DIR="$WORK_DIR/player${player}/video_parts"
    export OUTPUT_DIR="$WORK_DIR/player${player}/output"
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    export SCRIBBLE_STATIC_DIR="$SCRIPT_DIR/../static"
    
    echo -e "${YELLOW}"
    echo "=========================================="
    echo "🌐  WEBAPP INTERACTION REQUIRED"  
    echo "=========================================="
    echo "The gaze tracking script will open a web interface at:"
    echo "http://localhost:5000"
    echo ""
    echo "Please:"
    echo "1. Open the URL in your browser"
    echo "2. Mark the objects you want to track"
    echo "3. Click 'Send' to continue processing"
    echo "4. Return here and press Enter when done"
    echo "=========================================="
    echo -e "${NC}"
    
    # Start the tracking script
    log_info "Running gaze analysis script..."
    cd "$SCRIPT_DIR"
    python analyze_player_gaze.py
    
    # Check that gaze tracking completed successfully
    if [ -d "$WORK_DIR/player${player}/output/gaze_data" ]; then
        log_success "Player $player gaze tracking completed"
        log_info "Individual player data kept in workspace for sync analysis"
    else
        log_error "Player $player gaze tracking failed - no output data found"
        exit 1
    fi
}

run_synchronization_analysis() {
    local analysis_type=$1
    log_step "Running synchronization analysis for $analysis_type..."
    
    # Look for the CSV files from both players (gaze_hits_per_frame.csv specifically)
    # The files are in the workspace output directories, not the final output directories yet
    P1_CSV=$(find "$WORK_DIR/player1/output" -name "gaze_hits_per_frame.csv" -type f | head -1)
    P2_CSV=$(find "$WORK_DIR/player2/output" -name "gaze_hits_per_frame.csv" -type f | head -1)
    
    if [ -z "$P1_CSV" ] || [ -z "$P2_CSV" ]; then
        log_error "Could not find CSV files for both players"
        log_info "Looking in: $WORK_DIR/player1/output and $WORK_DIR/player2/output"
        log_info "Available files in player1 output:"
        find "$WORK_DIR/player1/output" -type f 2>/dev/null | head -10
        log_info "Available files in player2 output:"
        find "$WORK_DIR/player2/output" -type f 2>/dev/null | head -10
        exit 1
    fi
    
    log_info "Using Player 1 CSV: $(basename $P1_CSV)"
    log_info "Using Player 2 CSV: $(basename $P2_CSV)"
    
    # Copy CSV files to workspace for analysis
    cp "$P1_CSV" "$WORK_DIR/player1_gaze.csv"
    cp "$P2_CSV" "$WORK_DIR/player2_gaze.csv"
    
    # Run synchronization analysis
    cd "$WORK_DIR"
    log_info "Running synchronization analysis..."
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    python "$SCRIPT_DIR/sync_players_data.py" player1_gaze.csv player2_gaze.csv
    
    # Move results to output directory
    if [ -d "sync_results" ]; then
        # Ensure sync_results directory exists in output
        mkdir -p "$OUTPUT_DIR/sync_results"
        # Copy contents if sync_results is not empty
        if [ "$(ls -A sync_results)" ]; then
            cp -r sync_results/* "$OUTPUT_DIR/sync_results/"
        fi
        log_success "Synchronization analysis completed"
    else
        log_error "Synchronization analysis failed - no results generated"
        exit 1
    fi
    
    # Copy sync_analysis.csv for multi-condition analysis
    if [ -f "sync_analysis.csv" ]; then
        cp "sync_analysis.csv" "$OUTPUT_DIR/pair_01/sync_analysis.csv"
        log_info "Saved sync_analysis.csv for multi-condition analysis"
    fi
}

cleanup_workspace() {
    log_step "Cleaning up temporary files..."
    rm -rf "$WORK_DIR"
    log_success "Cleanup completed"
}

show_results() {
    echo -e "${GREEN}"
    echo "=================================================="
    echo "🎉  ANALYSIS COMPLETED SUCCESSFULLY!  🎉"
    echo "=================================================="
    echo -e "${NC}"
    echo "Results saved to host folder (mounted at $OUTPUT_DIR):"
    echo ""
    echo "📁 Player 1 Results: $OUTPUT_DIR/player1_results/"
    echo "📁 Player 2 Results: $OUTPUT_DIR/player2_results/" 
    echo "📊 Synchronization Graphs: $OUTPUT_DIR/sync_results/"
    echo ""
    echo "Graphs generated:"
    ls -1 "$OUTPUT_DIR/sync_results/" 2>/dev/null | sed 's/^/   📈 /'
    echo ""
    echo -e "${BLUE}Analysis pipeline completed successfully!${NC}"
}

main() {
    # Check arguments
    if [ $# -ne 3 ]; then
        echo "Usage: $0 <player1_video.mp4> <player2_video.mp4> <cooperation|competition>"
        echo ""
        echo "Example: $0 player1_recording.mp4 player2_recording.mp4 cooperation"
        echo ""
        echo "Videos should be placed in the host folder mounted to /app/input"
        exit 1
    fi
    
    local player1_video="$1"
    local player2_video="$2"
    local analysis_type="$3"
    
    # Validate analysis type
    if [[ "$analysis_type" != "cooperation" && "$analysis_type" != "competition" ]]; then
        log_error "Analysis type must be either 'cooperation' or 'competition'"
        exit 1
    fi
    
    print_banner
    
    # Main pipeline
    check_prerequisites "$player1_video" "$player2_video"
    setup_workspace
    
    # Process videos sequentially
    split_video 1 "$player1_video"
    split_video 2 "$player2_video"
    
    # Run gaze tracking for each player
    run_gaze_tracking 1 "$player1_video" "$analysis_type"
    run_gaze_tracking 2 "$player2_video" "$analysis_type"
    
    # Run synchronization analysis
    run_synchronization_analysis "$analysis_type"
    
    # Cleanup and show results
    cleanup_workspace
    show_results
}

# Run main function with all arguments
main "$@"