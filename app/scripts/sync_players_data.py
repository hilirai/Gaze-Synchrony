import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import os

def analyze_gaze_sync(player1_file, player2_file):
    """
    Analyze gaze synchronization between two players.
    
    Args:
        player1_file: CSV file for player 1 gaze data
        player2_file: CSV file for player 2 gaze data
        
    Returns:
        sync_results: List of synchronization data per frame
        obj_cols: List of object column names
        sync_df: DataFrame with sync analysis for saving
    """
    # Load the data
    p1_data = pd.read_csv(player1_file)
    p2_data = pd.read_csv(player2_file)
    
    # Check for and apply frame offsets
    p1_offset = 0
    p2_offset = 0
    
    # Try to find frame offset files relative to the CSV files
    p1_dir = os.path.dirname(player1_file)
    p2_dir = os.path.dirname(player2_file)
    
    p1_offset_file = os.path.join(p1_dir, "selected_frame_offset.txt")
    p2_offset_file = os.path.join(p2_dir, "selected_frame_offset.txt")
    
    if os.path.exists(p1_offset_file):
        with open(p1_offset_file, 'r') as f:
            p1_offset = int(f.read().strip())
        print(f"[info] Player 1 frame offset: {p1_offset}")
    
    if os.path.exists(p2_offset_file):
        with open(p2_offset_file, 'r') as f:
            p2_offset = int(f.read().strip())
        print(f"[info] Player 2 frame offset: {p2_offset}")
    
    # Apply frame offsets by adjusting frame numbers
    if p1_offset > 0:
        p1_data = p1_data.copy()
        p1_data['frame'] = p1_data['frame'] + p1_offset
        
    if p2_offset > 0:
        p2_data = p2_data.copy()
        p2_data['frame'] = p2_data['frame'] + p2_offset
    
    print(f"[info] Frame alignment applied - P1 offset: {p1_offset}, P2 offset: {p2_offset}")
    print(f"[info] P1 frame range: {p1_data['frame'].min()}-{p1_data['frame'].max()}")
    print(f"[info] P2 frame range: {p2_data['frame'].min()}-{p2_data['frame'].max()}")
    
    # Get object columns (excluding frame column)
    obj_cols = [col for col in p1_data.columns if col.startswith('obj_')]
    
    # Calculate synchronization for each frame
    sync_results = []
    
    # Get common frames between both players
    common_frames = set(p1_data['frame']) & set(p2_data['frame'])
    common_frames = sorted(list(common_frames))
    
    print(f"[info] Found {len(common_frames)} common frames between players")
    
    for frame in common_frames:
        p1_rows = p1_data[p1_data['frame'] == frame]
        p2_rows = p2_data[p2_data['frame'] == frame]
        
        # Skip if frame doesn't exist in either dataset
        if p1_rows.empty or p2_rows.empty:
            continue
            
        p1_row = p1_rows.iloc[0]
        p2_row = p2_rows.iloc[0]
        
        # Check if both players are looking at the same objects
        same_objects = []
        object_sync_status = {}
        
        for obj_col in obj_cols:
            if obj_col in p1_row and obj_col in p2_row:
                is_synced = p1_row[obj_col] == 1 and p2_row[obj_col] == 1
                object_sync_status[obj_col] = is_synced
                if is_synced:
                    same_objects.append(obj_col)
        
        sync_results.append({
            'frame': frame,
            'synchronized_objects': same_objects,
            'sync_count': len(same_objects),
            'object_sync_status': object_sync_status,
            'p1_looking_at': [obj for obj in obj_cols if obj in p1_row and p1_row[obj] == 1],
            'p2_looking_at': [obj for obj in obj_cols if obj in p2_row and p2_row[obj] == 1]
        })
    
    # Create DataFrame for CSV export
    sync_df_data = []
    for result in sync_results:
        row_data = {
            'frame': result['frame'],
            'sync_count': result['sync_count'],
            'synchronized_objects': ','.join(result['synchronized_objects']),
            'p1_looking_at': ','.join(result['p1_looking_at']),
            'p2_looking_at': ','.join(result['p2_looking_at'])
        }
        # Add individual object sync status
        for obj_col in obj_cols:
            row_data[f'{obj_col}_sync'] = result['object_sync_status'][obj_col]
        sync_df_data.append(row_data)
    
    sync_df = pd.DataFrame(sync_df_data)
    
    return sync_results, obj_cols, sync_df

def plot_synchronization(sync_results, obj_cols):
    """Create separate graphs for overall sync and individual object synchronization."""
    import os
    
    frames = [result['frame'] for result in sync_results]
    sync_counts = [result['sync_count'] for result in sync_results]
    
    # Create results directory
    results_dir = 'sync_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Overall synchronization summary
    plt.figure(figsize=(12, 6))
    plt.plot(frames, sync_counts, marker='o', linewidth=2, markersize=4, color='blue')
    plt.xlabel('Time Point')
    plt.ylabel('Synchronized Objects Count')
    
    # Calculate overall sync rate for title
    avg_sync = sum(sync_counts) / len(sync_counts)
    total_possible_syncs = len(frames) * len(obj_cols)  # Use actual object columns
    total_actual_syncs = sum(sync_counts)
    overall_sync_rate = (total_actual_syncs / total_possible_syncs) * 100
    
    plt.title(f'Gaze Synchronization Overview\nAvg: {avg_sync:.2f} objects synchronized per moment | Overall Sync Rate: {overall_sync_rate:.1f}%')
    plt.grid(True, alpha=0.3)
    plt.xticks(frames[::max(1, len(frames)//10)])
    
    # Highlight frames with high synchronization
    max_sync = max(sync_counts) if sync_counts else 0
    if max_sync > 0:
        for i, (frame, count) in enumerate(zip(frames, sync_counts)):
            if count == max_sync:
                plt.annotate(f'Peak: {count}', 
                            xy=(frame, count), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                break
    
# Summary statistics now shown in title
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'overall_synchronization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved overall synchronization graph to: {results_dir}/overall_synchronization.png")
    
    # Individual object synchronization plots
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    for i, obj_col in enumerate(obj_cols):
        plt.figure(figsize=(12, 6))
        
        # Extract synchronization status for this object across all frames
        obj_sync = [result['object_sync_status'][obj_col] for result in sync_results]
        obj_sync_binary = [1 if sync else 0 for sync in obj_sync]
        
        # Plot as step function for binary data
        plt.step(frames, obj_sync_binary, where='mid', linewidth=3, 
                color=colors[i % len(colors)], marker='o', markersize=4, label='Synchronized')
        plt.fill_between(frames, obj_sync_binary, step='mid', alpha=0.3, 
                       color=colors[i % len(colors)])
        
        plt.xlabel('Time Point')
        plt.ylabel('Players Looking at Same Object')
        
        # Calculate sync rate for title
        sync_rate = sum(obj_sync_binary) / len(obj_sync_binary) * 100
        plt.title(f'{obj_col.replace("_", " ").title()} - Joint Attention Analysis\nBoth Players Looking: {sync_rate:.1f}% of time')
        
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ['No Joint Attention', 'Joint Attention'])
        plt.grid(True, alpha=0.3)
        plt.xticks(frames[::max(1, len(frames)//10)])
        
        # Add detailed statistics
        sync_moments_count = sum(obj_sync_binary)
        total_moments = len(obj_sync_binary)
        
        # Find longest sync periods
        sync_periods = []
        current_period = 0
        for sync in obj_sync_binary:
            if sync:
                current_period += 1
            else:
                if current_period > 0:
                    sync_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            sync_periods.append(current_period)
        
        max_period = max(sync_periods) if sync_periods else 0
        avg_period = sum(sync_periods) / len(sync_periods) if sync_periods else 0
        
        stats_text = f'Joint attention: {sync_rate:.1f}% ({sync_moments_count}/{total_moments} moments)\n'
        stats_text += f'Longest joint attention: {max_period} consecutive moments\n'
        stats_text += f'Average joint attention duration: {avg_period:.1f} moments\n'
        stats_text += f'Number of joint attention episodes: {len(sync_periods)}'
        
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                verticalalignment='top')
        
        plt.tight_layout()
        filename = f'{obj_col}_synchronization.png'
        plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved {obj_col} synchronization graph to: {results_dir}/{filename}")
    
    print(f"\n[INFO] All graphs saved to directory: {results_dir}/")
    return frames, sync_counts

def print_detailed_results(sync_results):
    """Print detailed synchronization results."""
    print("=== Joint Attention Analysis Results ===\n")
    
    total_moments = len(sync_results)
    moments_with_sync = sum(1 for r in sync_results if r['sync_count'] > 0)
    avg_sync = sum(r['sync_count'] for r in sync_results) / total_moments
    
    print(f"Total time points analyzed: {total_moments}")
    print(f"Moments with joint attention: {moments_with_sync}")
    print(f"Joint attention occurrence rate: {moments_with_sync/total_moments:.2%}")
    print(f"Average objects with joint attention per moment: {avg_sync:.2f}")
    print()
    
    print("Moment-by-moment analysis:")
    print("-" * 70)
    for result in sync_results:
        time_point = result['frame']
        joint_objects = result['synchronized_objects']
        p1_objects = result['p1_looking_at']
        p2_objects = result['p2_looking_at']
        
        if joint_objects:
            print(f"Point {time_point:2d}: JOINT ATTENTION on {', '.join(joint_objects)} | P1: {', '.join(p1_objects)} | P2: {', '.join(p2_objects)}")
        else:
            print(f"Point {time_point:2d}: No joint attention | P1: {', '.join(p1_objects)} | P2: {', '.join(p2_objects)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python sync_players_data.py <player1_gaze.csv> <player2_gaze.csv>")
        sys.exit(1)
    
    player1_file = sys.argv[1]
    player2_file = sys.argv[2]
    
    print(f"Starting gaze synchronization analysis...")
    print(f"Player 1 data: {player1_file}")
    print(f"Player 2 data: {player2_file}")
    
    try:
        sync_results, obj_cols, sync_df = analyze_gaze_sync(player1_file, player2_file)
        
        # Save sync analysis as CSV
        sync_csv_path = os.path.join(os.path.dirname(player1_file), "sync_analysis.csv")
        sync_df.to_csv(sync_csv_path, index=False)
        print(f"[OK] Saved synchronization analysis to: {sync_csv_path}")
        
        print_detailed_results(sync_results)
        plot_synchronization(sync_results, obj_cols)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)