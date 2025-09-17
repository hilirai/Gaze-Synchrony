# Gaze Synchronization Pipeline Overview

## Pipeline Architecture

The gaze synchronization analysis pipeline consists of three main phases that process eye-tracking videos to detect synchronized attention between participants.

### Phase A: Object Selection & Tracking

**Purpose**: Initialize SAM2 tracking for objects of interest (AOIs)

**Process**:
1. **Frame Extraction**: Extract 50 frames from the first video segment for user selection
2. **Interactive Selection**: 
   - Web interface displays frames with coordinate grid
   - User draws green strokes on objects to track
   - User draws red strokes on areas to avoid (optional)
   - User selects the clearest frame for tracking initialization
3. **Stroke Processing**: Convert user strokes into seed points using uniform sampling
4. **SAM2 Initialization**: Initialize video predictor with positive/negative seed points
5. **Object Tracking**: Propagate segmentation masks across all video frames
6. **Mask Export**: Save bounding boxes for each tracked object per frame

**Key Files**:
- `src/analyze_player_gaze.py` - Main tracking implementation
- `app/app.py` - Flask web interface for selection
- `app/templates/scribble.html` - Drawing interface

**Outputs**:
- `track_file.csv` - Object bounding boxes per frame
- `overlays/` - Visualization images (first 10 frames)

### Phase B: Gaze-Object Mapping

**Purpose**: Map detected gaze points to tracked objects

**Process**:
1. **Gaze Detection**: 
   - HSV color segmentation to find red gaze dots
   - Morphological operations to clean detection
   - Centroid calculation with radius estimation
2. **Hit Detection**: 
   - For each frame, check if gaze point intersects object bounding boxes
   - Apply configurable gaze radius for tolerance
   - Record hit/miss for each object
3. **Data Export**: Generate both long-form and wide-form CSV files

**Algorithm**:
```python
for each frame:
    gaze_point = detect_red_dot(frame)
    for each tracked_object:
        if gaze_point within (bbox + gaze_radius):
            record_hit(frame, object_id, True)
        else:
            record_hit(frame, object_id, False)
```

**Outputs**:
- `gaze_hits.csv` - Long-form: frame, object_id, hit (boolean)  
- `gaze_hits_per_frame.csv` - Wide-form: frame, obj_0, obj_1, obj_2, ...

### Phase C: Synchronization Analysis

**Purpose**: Analyze gaze synchrony between participants

**Process**:
1. **Frame Alignment**: Apply frame offsets from user-selected seed frames
2. **Synchrony Detection**: 
   - Find common frames between both participants
   - Check if both participants fixate the same object(s)
   - Count synchronized objects per frame
3. **Statistical Analysis**:
   - Calculate synchrony rates per object
   - Compute episode durations and frequencies
   - Generate condition comparisons
4. **Visualization**: Create timeline plots and histograms

**Synchrony Definition**:
Participants are synchronized when both fixate the same AOI within a 1-frame (~33ms) temporal window.

**Key Files**:
- `src/sync_players_gaze.py` - Pairwise synchrony analysis
- `src/generate_results_graphs.py` - Multi-condition statistics

**Outputs**:
- `sync_analysis.csv` - Per-frame synchrony data
- `synchronization_histogram.png` - Condition comparison
- `timeline_probabilities.png` - Synchrony over time

## AOI (Areas of Interest) Schema

| AOI Label | Description | Typical Objects |
|-----------|-------------|-----------------|
| `hand_self` | Hands of participant wearing eye-tracker | Player's own hands |
| `hand_partner` | Other participant's hands | Partner's hands |
| `die` | Game die | Dice used in task |
| `tower` | Central structure | Game board, base, support |
| `game_pieces` | Movable elements | Cards, tokens, pieces |

## Data Flow

```
Raw Videos → Phase A → track_file.csv → Phase B → gaze_hits.csv → Phase C → sync_analysis.csv
    ↓            ↓                          ↓                         ↓
Video Segments   Overlays               Wide Format               Visualizations
```

## Configuration Parameters

### Critical Parameters (config.yaml)

- `gaze.radius`: Detection tolerance (default: 100 pixels)
- `model.memory_fraction`: GPU memory usage (default: 0.9)
- `tracking.sampling.strategy`: Point sampling method ("uniform" recommended)
- `synchronization.time_window`: Temporal synchrony window (default: 1 frame)

### Performance Tuning

- **GPU Memory Issues**: Reduce `model.memory_fraction` or enable `processing.single_process`
- **Speed Optimization**: Adjust `tracking.memory_cleanup_interval`
- **Detection Accuracy**: Fine-tune `gaze.hsv_ranges` for different lighting conditions

## Validation & Quality Control

### Automatic Validation
- Frame count verification across video segments
- Gaze detection rate monitoring  
- Object tracking consistency checks
- Synchrony rate bounds checking

### Manual Validation
- Overlay image inspection for tracking quality
- Gaze detection accuracy on sample frames
- Synchrony timeline visual inspection
- Cross-condition consistency validation

## Technical Specifications

- **Video Processing**: OpenCV + ffmpeg
- **Object Tracking**: SAM2 (Segment Anything Model 2)
- **Gaze Detection**: HSV color segmentation
- **Synchrony Metric**: Binary hit detection with temporal windowing
- **Statistical Analysis**: Pandas + NumPy + SciPy
- **Visualization**: Matplotlib

## Performance Characteristics

- **Processing Speed**: ~1-2 minutes per minute of video (GPU)
- **Memory Usage**: ~8GB GPU RAM for typical videos
- **Accuracy**: >95% gaze detection under good lighting
- **Scalability**: Handles videos up to 30 minutes per participant

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor gaze detection | Lighting/camera setup | Adjust HSV ranges in config |
| GPU out of memory | Large videos/limited VRAM | Reduce memory_fraction or use CPU |
| Web interface timeout | Slow user selection | Increase timeout in Flask config |
| Synchrony rate too low/high | Misaligned videos | Check frame offset calculation |

## Research Applications

This pipeline supports analysis of:
- **Cooperative vs. Competitive Tasks**: Different synchrony patterns
- **Task Complexity Effects**: Synchrony changes with difficulty
- **Individual Differences**: Personality/skill impact on synchrony
- **Temporal Dynamics**: How synchrony evolves during tasks
- **Object-Specific Patterns**: Which AOIs drive synchrony