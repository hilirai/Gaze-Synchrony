import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sam2.build_sam import build_sam2_video_predictor
import requests
import webbrowser
import subprocess
import threading
import time

# ====== CONFIG ======
CHECKPOINT  = "/home/administrator/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG   = "/configs/sam2.1/sam2.1_hiera_b+.yaml"
VIDEO_DIR   = os.environ.get("VIDEO_DIR", "/sam2/Gaze-Synchrony/app/data/player_1/video_parts")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "/sam2/Gaze-Synchrony/app/data/player_1/output")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
GAZE_RADIUS = 100  # pixels
MAX_SAVE_OVERLAY = 250
# Configure PyTorch CUDA memory management
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    torch.cuda.empty_cache()
    print(f"[info] GPU memory configured: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Shared with Flask: where it serves /static from (env overrides)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCRIBBLE_STATIC_DIR = os.environ.get("SCRIBBLE_STATIC_DIR", os.path.join(SCRIPT_DIR, "../app/static"))
os.makedirs(SCRIBBLE_STATIC_DIR, exist_ok=True)

# exposed to worker
PARTS = []

# ====== HELPERS ======
def detect_gaze_point(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l1, u1 = np.array([0,70,50]), np.array([10,255,255])
    l2, u2 = np.array([170,70,50]), np.array([180,255,255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    
    # Calculate center coordinates
    center_x = int(M["m10"]/M["m00"])
    center_y = int(M["m01"]/M["m00"])
    
    # Calculate radius from contour area (assuming circular shape)
    area = cv2.contourArea(c)
    radius = int(np.sqrt(area / np.pi)) if area > 0 else GAZE_RADIUS
    
    return center_x, center_y, radius

def draw_numbered_grid(image, spacing=50):
    h, w = image.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(image, (x,0), (x,h), (200,200,200), 1)
        cv2.putText(image, str(x), (x+2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (225,225,225), 1)
    for y in range(0, h, spacing):
        cv2.line(image, (0,y), (w,y), (200,200,200), 1)
        pos = y-2 if y>12 else y+12
        cv2.putText(image, str(y), (2,pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (225,225,225), 1)

def overlay_mask(image, mask):
    overlay = image.copy()
    m = mask.squeeze().cpu().numpy()
    bin_mask = (m > 0.5).astype(np.uint8)*255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
    return overlay

# ====== WORKER (optimized for speed) ======
def process_gaze_group(item):
    """
    Process one video part's rows, reading frames SEQUENTIALLY (no random seeks).
    item = (part_idx, rows_dataframe)
    """
    part_idx, rows = item
    part_path = PARTS[part_idx]
    rows = rows.sort_values("local_frame").reset_index(drop=True)

    cap = cv2.VideoCapture(part_path)
    if not cap.isOpened():
        return []

    results = []
    next_frame_idx = 0
    frame_buf = None

    for _, row in rows.iterrows():
        lf = int(row.local_frame)

        if lf < next_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, lf)
            next_frame_idx = lf

        while next_frame_idx <= lf:
            ret, frame_buf = cap.read()
            if not ret:
                break
            next_frame_idx += 1

        if frame_buf is None:
            results.append({'frame': int(row.global_frame),
                            'object_id': int(row.object_id),
                            'hit': False})
            continue

        if not (pd.notnull(row.xmin) and pd.notnull(row.ymin) and pd.notnull(row.xmax) and pd.notnull(row.ymax)):
            results.append({'frame': int(row.global_frame),
                            'object_id': int(row.object_id),
                            'hit': False})
            continue

        g = detect_gaze_point(frame_buf)
        hit = False
        if g:
            gx, gy, gaze_radius = g
            if (row.xmin - gaze_radius <= gx <= row.xmax + gaze_radius and
                row.ymin - gaze_radius <= gy <= row.ymax + gaze_radius):
                hit = True

        results.append({'frame': int(row.global_frame),
                        'object_id': int(row.object_id),
                        'hit': hit})

    cap.release()
    return results

# ====== MAIN ======
def main():
    global PARTS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "overlays"), exist_ok=True)
    print("video dir", VIDEO_DIR)
    parts = sorted(glob.glob(os.path.join(VIDEO_DIR, "video_part_*.mp4")))
    parts = parts[:2]
    if not parts:
        raise RuntimeError(f"No video_part_*.mp4 in {VIDEO_DIR}")
    offsets, cum = [], 0
    for p in parts:
        cap = cv2.VideoCapture(p)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        offsets.append(cum)
        cum += cnt

    # 2) seeding grid
    # Extract first 10 frames for selection
    cap = cv2.VideoCapture(parts[0])
    frames_extracted = 0
    frame_files = []
    
    print(f"[info] Extracting first 10 frames for selection...")
    
    for frame_idx in range(50):
        ret, frame = cap.read()
        if not ret:
            print(f"[warning] Could only extract {frames_extracted} frames from video")
            break
            
        # Add numbered grid to frame
        draw_numbered_grid(frame)
        
        # Save to both OUTPUT_DIR and Flask static folder
        grid_filename = f"seed_grid_frame_{frame_idx}.jpg"
        grid_path = os.path.join(OUTPUT_DIR, grid_filename)
        cv2.imwrite(grid_path, frame)
        
        static_grid = os.path.join(SCRIBBLE_STATIC_DIR, grid_filename)
        cv2.imwrite(static_grid, frame)
        
        frame_files.append(grid_filename)
        frames_extracted += 1
        
    cap.release()
    
    # Also save the first frame as the default seed_grid.jpg for backwards compatibility
    if frames_extracted > 0:
        default_grid_path = os.path.join(OUTPUT_DIR, "seed_grid.jpg")
        default_static_grid = os.path.join(SCRIBBLE_STATIC_DIR, "seed_grid.jpg")
        
        # Copy first frame as default
        import shutil
        shutil.copy(os.path.join(OUTPUT_DIR, "seed_grid_frame_0.jpg"), default_grid_path)
        shutil.copy(os.path.join(SCRIBBLE_STATIC_DIR, "seed_grid_frame_0.jpg"), default_static_grid)
    
    print(f"[info] Extracted {frames_extracted} frames: {frame_files}")
    print(f"[info] Default grid saved as seed_grid.jpg")

    # Start Flask server in background
    def start_flask_server():
        app_path = os.path.join(os.path.dirname(__file__), "../app.py")
        subprocess.run(["python", app_path], 
                      env=dict(os.environ, SCRIBBLE_STATIC_DIR=SCRIBBLE_STATIC_DIR))
    
    print("[info] Starting web interface server...")
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Wait for server to start
    print("[info] Waiting for server to start...")
    for _ in range(10):  # Wait up to 10 seconds
        try:
            resp = requests.get("http://localhost:5000", timeout=1)
            if resp.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    
    # NEW scribble-based seeding:
    print("\n📸 Open http://localhost:5000 — draw green (pos) and optional red (neg), then click Send.")
    print("🖼️  Use Previous/Next buttons to select the clearest frame!")
    try:
        webbrowser.open("http://localhost:5000")
    except Exception:
        pass
    input("Press Enter here after clicking Send in the browser… ")

    # Fetch strokes from Flask
    resp = requests.get("http://localhost:5000/get", timeout=10)
    resp.raise_for_status()
    response = resp.json()
    
    # Handle both old and new format
    if "strokes" in response:
        strokes = response["strokes"]
        selected_frame = response.get("selectedFrame", 0)
    else:
        # Legacy format
        strokes = response
        selected_frame = 0
    
    print(f"[info] User selected frame {selected_frame} for analysis")
    
    # Save the selected frame offset to file
    frame_offset_file = os.path.join(OUTPUT_DIR, "selected_frame_offset.txt")
    with open(frame_offset_file, 'w') as f:
        f.write(str(selected_frame))
    print(f"[info] Saved frame offset {selected_frame} to {frame_offset_file}")

    # # Turn positive strokes into seed points (centroids)
    # seeds = []
    # for stroke in strokes.get("pos", []):
    #     if not stroke:
    #         continue
    #     xs, ys = zip(*stroke)
    #     seeds.append((int(sum(xs)/len(xs)), int(sum(ys)/len(ys))))
    
    # # Turn negative strokes into negative seed points (centroids)
    # neg_seeds = []
    # for stroke in strokes.get("neg", []):
    #     if not stroke:
    #         continue
    #     xs, ys = zip(*stroke)
    #     neg_seeds.append((int(sum(xs)/len(xs)), int(sum(ys)/len(ys))))
    
    # n_objs = len(seeds)
    # n_neg = len(neg_seeds)
    # print(f"Loaded {n_objs} positive seed points and {n_neg} negative seed points from scribbles.")

    # --- Multi-point sampling along strokes (more robust than a single centroid) ---
    # Configurable sampling (env vars), with sensible defaults
    POS_SAMPLING_STEP = int(os.environ.get("POS_SAMPLING_STEP", "8"))    # take every Nth vertex of a positive stroke
    POS_SAMPLING_CAP  = int(os.environ.get("POS_SAMPLING_CAP",  "12"))   # limit max positive points per stroke
    NEG_SAMPLING_STEP = int(os.environ.get("NEG_SAMPLING_STEP", "8"))    # take every Nth vertex of a negative stroke
    NEG_SAMPLING_CAP  = int(os.environ.get("NEG_SAMPLING_CAP",  "5"))   # limit max negative points total
    SAMPLING_STRATEGY = "uniform" 
    def _sample_points(stroke, step, cap):
        """
        Return up to `cap` points from `stroke` according to SAMPLING_STRATEGY.
        - "stride": take every `step`-th vertex, then first `cap`.
        - "uniform": sample *interior* points by arc-length (exclude endpoints),
                    enforce min spacing, and cap to a small count (3–5 is ideal).
        """
        if not stroke:
            return []

        if SAMPLING_STRATEGY == "stride":
            pts = stroke[::max(1, step)]
            pts = pts[:cap]
            return [[int(x), int(y)] for x, y in pts]

        # --- uniform (SAM-friendly) ---
        xs = np.asarray([p[0] for p in stroke], dtype=np.float32)
        ys = np.asarray([p[1] for p in stroke], dtype=np.float32)

        if xs.size == 1:
            return [[int(xs[0]), int(ys[0])]]

        dx = np.diff(xs); dy = np.diff(ys)
        seg = np.hypot(dx, dy)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total == 0.0:
            return [[int(xs[0]), int(ys[0])]]

        # Prefer a *small* number of interior positives
        # (SAM2 usually works best with ~3–5)
        hard_cap = int(min(cap, 5))
        if hard_cap <= 1:
            # pick a robust "middle" point
            mid_t = total * 0.5
            j = int(np.searchsorted(cum, mid_t, side="right"))
            j = max(1, min(j, cum.size - 1))
            t0, t1 = cum[j-1], cum[j]
            alpha = 0.0 if t1 == t0 else (mid_t - t0) / (t1 - t0)
            xi = xs[j-1] + alpha * (xs[j] - xs[j-1])
            yi = ys[j-1] + alpha * (ys[j] - ys[j-1])
            return [[int(round(xi)), int(round(yi))]]

        # Exclude endpoints (avoid borders) and bias toward the interior band
        # e.g., 10%..90% of the arc length
        interior_lo, interior_hi = 0.10, 0.90
        targets = np.linspace(total * interior_lo, total * interior_hi, num=hard_cap)

        pts = []
        last_accepted = None
        min_sep = 6.0  # px; increase to 8–10 if you still see clumps

        j = 1
        for t in targets:
            # find segment
            while j < cum.size and cum[j] < t:
                j += 1
            j = min(j, cum.size - 1)
            t0, t1 = cum[j-1], cum[j]
            if t1 == t0:
                xi, yi = xs[j], ys[j]
            else:
                alpha = (t - t0) / (t1 - t0)
                xi = xs[j-1] + alpha * (xs[j] - xs[j-1])
                yi = ys[j-1] + alpha * (ys[j] - ys[j-1])

            # spacing filter
            if last_accepted is None or np.hypot(xi - last_accepted[0], yi - last_accepted[1]) >= min_sep:
                pts.append([xi, yi])
                last_accepted = (xi, yi)

        # de-dup after rounding
        out = []
        seen = set()
        for xi, yi in pts:
            xi_i, yi_i = int(round(xi)), int(round(yi))
            key = (xi_i, yi_i)
            if key in seen:
                continue
            seen.add(key)
            out.append([xi_i, yi_i])

        # ensure at least one robust point (center) if everything deduped away
        if not out:
            mid_t = total * 0.5
            j = int(np.searchsorted(cum, mid_t, side="right"))
            j = max(1, min(j, cum.size - 1))
            t0, t1 = cum[j-1], cum[j]
            alpha = 0.0 if t1 == t0 else (mid_t - t0) / (t1 - t0)
            xi = xs[j-1] + alpha * (xs[j] - xs[j-1])
            yi = ys[j-1] + alpha * (ys[j] - ys[j-1])
            out = [[int(round(xi)), int(round(yi))]]

        return out


    # For positives: one list of points **per object/stroke**
    pos_seeds = []
    for stroke in strokes.get("pos", []):
        if not stroke:
            continue
        pos_seeds.append(_sample_points(stroke, POS_SAMPLING_STEP, POS_SAMPLING_CAP))

    # Keep centroids as a fallback (used later for reseeding between parts)
    seed_centroids = []
    for pts in pos_seeds:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx = int(sum(xs)/len(xs)); cy = int(sum(ys)/len(ys))
        seed_centroids.append((cx, cy))

    # For negatives: flatten all sampled points into one shared list
    neg_points = []
    for stroke in strokes.get("neg", []):
        if not stroke:
            continue
        neg_points.extend(_sample_points(stroke, NEG_SAMPLING_STEP, NEG_SAMPLING_CAP))

    n_objs = len(pos_seeds)
    total_pos_pts = sum(len(pts) for pts in pos_seeds)
    n_neg = len(neg_points)
    print(f"Loaded {n_objs} objects from scribbles "
          f"({total_pos_pts} positive points total) and {n_neg} negative points.")

    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT)
    track_records = []

    for idx, part in enumerate(parts):
        print(f"\nPhase A: Tracking part {idx} → {part}")
        cap = cv2.VideoCapture(part)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=torch.float16):
            state = predictor.init_state(part)
            # for oid, (sx, sy) in enumerate(seeds):
            #     # Add positive seed point
            #     pos_points = [[sx, sy]]
            #     pos_labels = [1]
                
            #     # Add negative seed points if they exist
            #     neg_points = [[nx, ny] for nx, ny in neg_seeds]
            #     neg_labels = [0] * len(neg_seeds)
                
            #     # Combine positive and negative points
            #     all_points = pos_points + neg_points
            #     all_labels = pos_labels + neg_labels
                
            #     predictor.add_new_points_or_box(
            #         state, frame_idx=selected_frame,
            #         points=all_points, labels=all_labels, obj_id=oid
            #     )

            for oid, pos_pts in enumerate(pos_seeds):
                # Positive prompts: multiple points per object
                pos_labels = [1] * len(pos_pts)
                # Negative prompts: shared "avoid" hints
                neg_labels = [0] * len(neg_points)
                # Combine prompts
                all_points = pos_pts + neg_points
                all_labels = pos_labels + neg_labels

                predictor.add_new_points_or_box(
                    state,
                    frame_idx=selected_frame,
                    points=all_points,
                    labels=all_labels,
                    obj_id=oid
                )

            last_masks = {}

            frame_counter = 0
            for lf, object_ids, masks in tqdm(
                predictor.propagate_in_video(state),
                desc=f"Part {idx} propagation",
                total=frame_count
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, lf)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Periodic memory cleanup every 50 frames
                frame_counter += 1
                if frame_counter % 50 == 0:
                    torch.cuda.empty_cache()

                for oid, mask in zip(object_ids, masks):
                    m = mask.squeeze().cpu().numpy()
                    ys, xs = np.where(m > 0.5)
                    xmin, xmax = (xs.min(), xs.max()) if xs.size else (None, None)
                    ymin, ymax = (ys.min(), ys.max()) if ys.size else (None, None)

                    track_records.append({
                        'part_idx': idx,
                        'local_frame': lf,
                        'global_frame': offsets[idx] + lf,
                        'object_id': oid,
                        'xmin': xmin, 'ymin': ymin,
                        'xmax': xmax, 'ymax': ymax
                    })

                    overlay = overlay_mask(frame, mask)
                    if (lf < MAX_SAVE_OVERLAY):
                        fname = f"part{idx:02d}_obj{oid}_f{lf:04d}.png"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, "overlays", fname), overlay)
                    last_masks[oid] = mask

        cap.release()

        # Clear GPU memory after each part
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        del state
        import gc
        gc.collect()
        print(f"[info] Cleared GPU memory after part {idx}")

        # reseed for next part
        if idx < len(parts) - 1:
            new_centroids = []
            for oid in range(n_objs):
                lm = last_masks.get(oid)
                if lm is not None:
                    ys, xs = np.where(lm.squeeze().cpu().numpy() > 0.5)
                    if xs.size:
                        # new_seeds.append((int(xs.mean()), int(ys.mean())))
                        new_centroids.append((int(xs.mean()), int(ys.mean())))
                    else:
                        # new_seeds.append(seeds[oid])
                        new_centroids.append(seed_centroids[oid])
                else:
                    new_centroids.append(seed_centroids[oid])

            # For the next part, prompt with **one** stable point per object (centroid)
            pos_seeds = [[[cx, cy]] for (cx, cy) in new_centroids]
            seed_centroids = new_centroids  # update fallback

    # Apply frame offset to global frame numbers
    frame_offset_file = os.path.join(OUTPUT_DIR, "selected_frame_offset.txt")
    frame_offset = 0
    if os.path.exists(frame_offset_file):
        with open(frame_offset_file, 'r') as f:
            frame_offset = int(f.read().strip())
        print(f"[info] Seed frame (no shift applied): {frame_offset}")
        
        # Adjust global frame numbers
        for record in track_records:
            record['global_frame'] = record['global_frame']
            record['frame_offset'] = frame_offset

    # save tracking CSV
    track_df = pd.DataFrame(track_records)
    track_csv = os.path.join(OUTPUT_DIR, "track_file.csv")
    track_df.to_csv(track_csv, index=False)
    print(f"\nPhase A complete: saved → {track_csv}")

    # 4) Phase B: parallel gaze mapping
    PARTS = parts
    track_df = pd.read_csv(track_csv)
    grouped = [(pid, df.reset_index(drop=True)) for pid, df in track_df.groupby('part_idx')]

    total_rows = sum(len(df) for _, df in grouped)
    print(f"Phase B: {len(grouped)} parts, {total_rows} rows to evaluate.")

    gaze_results = []

    if os.environ.get("GAZE_SINGLE_PROCESS") == "1":
        for item in tqdm(grouped, total=len(grouped), desc="Phase B parts (single)"):
            gaze_results.extend(process_gaze_group(item))
    else:
        with ProcessPoolExecutor(max_workers=min(len(grouped), os.cpu_count() or 1)) as pool:
            futures = {pool.submit(process_gaze_group, item): item[0] for item in grouped}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase B parts"):
                pid = futures[fut]
                try:
                    part_res = fut.result()
                    gaze_results.extend(part_res)
                except Exception as e:
                    print(f"[WARN] Phase B: part {pid} failed with: {e}")

    gaze_df = pd.DataFrame(gaze_results)
    gaze_df.sort_values(['frame','object_id'], inplace=True)

    # ---- NEW: per-object True/False per frame (wide format) ----
    gaze_df['hit'] = gaze_df['hit'].astype(bool)
    wide = gaze_df.pivot(index='frame', columns='object_id', values='hit').fillna(False)
    # name columns as obj_0, obj_1, ...
    wide.columns = [f'obj_{int(c)}' for c in wide.columns]
    wide = wide.reset_index().sort_values('frame')

    # ensure gaze_data dir exists
    gaze_dir = os.path.join(OUTPUT_DIR, "gaze_data")
    os.makedirs(gaze_dir, exist_ok=True)

    # write both files
    gaze_csv_long = os.path.join(gaze_dir, "gaze_hits.csv")
    gaze_csv_wide = os.path.join(gaze_dir, "gaze_hits_per_frame.csv")
    gaze_df.to_csv(gaze_csv_long, index=False)   # long (as before)
    wide.to_csv(gaze_csv_wide, index=False)      # wide with True/False per object

    # Optional console print per frame: set PRINT_GAZE_PER_FRAME=1 to enable
    if os.environ.get("PRINT_GAZE_PER_FRAME") == "1":
        for _, row in wide.iterrows():
            frame = int(row['frame'])
            status = {k: bool(row[k]) for k in row.index if k != 'frame'}
            print(f"[Gaze] frame {frame}: {status}")

    print(f"\nPhase B complete: saved → {gaze_csv_long} and {gaze_csv_wide}")

if __name__ == "__main__":
    main()
