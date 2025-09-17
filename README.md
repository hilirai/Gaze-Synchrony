# Neural and Gaze Synchrony During Cooperation and Competition

A pipeline for analyzing gaze synchronization between participants during cooperative and competitive tasks using SAM2 video object tracking and red-dot gaze detection.

## Overview

This project analyzes gaze synchronization patterns by:
1. **Phase A**: Interactive object selection and SAM2 video tracking
2. **Phase B**: Gaze-to-object mapping and synchronization analysis  
3. **Results**: Multi-condition statistical analysis and visualization

## Project Structure

```
├── app/                          # Web UI for object selection
│   ├── app.py                    # Flask backend
│   └── templates/scribble.html   # Drawing interface
├── src/                          # Analysis code
│   ├── analyze_player_gaze.py    # Per-player AOI detection
│   ├── sync_players_gaze.py      # Pairwise synchronization
│   └── generate_results_graphs.py # Statistical analysis
├── scripts/                      # Pipeline orchestration
│   ├── run_gaze_analysis.sh      # Main pipeline
│   └── start_analysis.sh         # Setup script
├── data/                         # Data directories
│   ├── raw/                      # Input videos
│   └── processed/                # Analysis outputs
└── docs/                         # Documentation
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Linux environment

### Quick Setup
```bash
git clone https://github.com/your-username/Gaze-Synchrony.git
cd Gaze-Synchrony
./setup.sh
```

### Manual Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Download SAM2 checkpoints
./download_ckpts.sh
```

## Usage

### Basic Pipeline
```bash
# Run full analysis for two participants
cd scripts
./run_gaze_analysis.sh player1_video.mp4 player2_video.mp4 cooperation
```

### Analysis Steps
1. **Object Selection**: Web interface opens at http://localhost:5000
   - Draw green strokes on objects to track
   - Draw red strokes to avoid (optional)
   - Select best frame and click "Send"

2. **Automatic Processing**: 
   - SAM2 tracks objects across video frames
   - Detects red gaze dots in each frame
   - Maps gaze to objects and calculates synchrony

3. **Results**: Generated in `data/processed/`
   - `gaze_csv/`: Per-player gaze data
   - `sync_csv/`: Synchronization analysis
   - `figures/`: Visualization plots

### Multi-Condition Analysis
```bash
cd src
python generate_results_graphs.py ../data/processed/
```

## Configuration

Edit `config.yaml` to customize:
- Video processing parameters
- Gaze detection thresholds  
- Output directories
- SAM2 model settings

## AOI Definitions

- **hand_self**: Participant wearing eye-tracker's hands
- **hand_partner**: Other participant's hands  
- **die**: Game die
- **tower**: Central tower/base structure
- **game_pieces**: Movable game elements

## Output Files

- `gaze_hits.csv`: Frame-by-frame gaze-object intersections
- `sync_analysis.csv`: Synchronization statistics per frame
- `synchronization_histogram.png`: Condition comparison
- `timeline_probabilities.png`: Synchrony over time

## Troubleshooting

### Common Issues
- **GPU Memory Error**: Reduce batch size or use `GAZE_SINGLE_PROCESS=1`
- **Web Interface Not Loading**: Check port 5000 availability
- **Missing Dependencies**: Run `./setup.sh` again

### Debug Mode
```bash
export PRINT_GAZE_PER_FRAME=1  # Enable frame-by-frame logging
export GAZE_SINGLE_PROCESS=1   # Disable multiprocessing
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gaze-synchrony-2024,
  title={Neural and Gaze Synchrony During Cooperation and Competition},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-username/Gaze-Synchrony}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SAM2 (Segment Anything Model 2) by Meta AI
- OpenCV for computer vision processing
- Flask for web interface