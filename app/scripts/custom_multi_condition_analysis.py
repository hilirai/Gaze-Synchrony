#!/usr/bin/env python3
"""
Custom Multi-Condition Gaze Synchronization Analysis
Generates histogram and timeline graphs for existing gaze synchronization data.

This script works with the current data structure and creates:
1. Histogram: Average synchronization percentage by condition
2. Timeline: Synchronization probability over time for each condition
3. AOI breakdown by condition and pair
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

def load_gaze_data_from_workspace(workspace_dir):
    """
    Load gaze synchronization data from workspace directory.
    
    Returns:
        tuple: (p1_gaze_df, p2_gaze_df) or (None, None) if not found
    """
    p1_csv = os.path.join(workspace_dir, "player1_gaze.csv")
    p2_csv = os.path.join(workspace_dir, "player2_gaze.csv")
    
    if os.path.exists(p1_csv) and os.path.exists(p2_csv):
        try:
            p1_df = pd.read_csv(p1_csv)
            p2_df = pd.read_csv(p2_csv)
            print(f"[OK] Loaded workspace data: P1={len(p1_df)} frames, P2={len(p2_df)} frames")
            return p1_df, p2_df
        except Exception as e:
            print(f"[WARN] Failed to load workspace data: {e}")
    else:
        print(f"[WARN] Workspace data not found: {p1_csv}, {p2_csv}")
    
    return None, None

def analyze_synchronization_from_gaze_data(p1_df, p2_df):
    """
    Analyze synchronization from player gaze data files.
    
    Returns:
        tuple: (sync_results, obj_cols)
    """
    # Get object columns (excluding frame column)
    obj_cols = [col for col in p1_df.columns if col.startswith('obj_')]
    
    print(f"[INFO] Found {len(obj_cols)} objects: {obj_cols}")
    
    # Get common frames between both players
    common_frames = set(p1_df['frame']) & set(p2_df['frame'])
    common_frames = sorted(list(common_frames))
    
    print(f"[INFO] Analyzing {len(common_frames)} common frames")
    
    sync_results = []
    
    for frame in common_frames:
        p1_row = p1_df[p1_df['frame'] == frame]
        p2_row = p2_df[p2_df['frame'] == frame]
        
        if p1_row.empty or p2_row.empty:
            continue
            
        p1_row = p1_row.iloc[0]
        p2_row = p2_row.iloc[0]
        
        # Check synchronization for each object
        same_objects = []
        object_sync_status = {}
        
        for obj_col in obj_cols:
            if obj_col in p1_row and obj_col in p2_row:
                # Both looking at same object (both True)
                is_synced = bool(p1_row[obj_col]) and bool(p2_row[obj_col])
                object_sync_status[obj_col] = is_synced
                if is_synced:
                    same_objects.append(obj_col)
        
        sync_results.append({
            'frame': frame,
            'synchronized_objects': same_objects,
            'sync_count': len(same_objects),
            'object_sync_status': object_sync_status,
            'p1_looking_at': [obj for obj in obj_cols if obj in p1_row and bool(p1_row[obj])],
            'p2_looking_at': [obj for obj in obj_cols if obj in p2_row and bool(p2_row[obj])]
        })
    
    return sync_results, obj_cols

def load_all_conditions_data(base_results_dir, workspace_dir=None):
    """
    Load synchronization data for all available conditions.
    
    Args:
        base_results_dir: Path to results directory
        workspace_dir: Path to workspace directory (for current session data)
        
    Returns:
        dict: {condition: {'pair_01': sync_results, ...}, ...}
    """
    conditions_data = {}
    
    # Check for existing results directories
    if os.path.exists(base_results_dir):
        for condition_dir in os.listdir(base_results_dir):
            condition_path = os.path.join(base_results_dir, condition_dir)
            if os.path.isdir(condition_path):
                print(f"[INFO] Found condition directory: {condition_dir}")
                # For now, we'll mark this as available but need the workspace data
                conditions_data[condition_dir] = {}
    
    # Load current workspace data if available
    if workspace_dir and os.path.exists(workspace_dir):
        p1_df, p2_df = load_gaze_data_from_workspace(workspace_dir)
        if p1_df is not None and p2_df is not None:
            sync_results, obj_cols = analyze_synchronization_from_gaze_data(p1_df, p2_df)
            
            # Determine condition based on available results directories
            # Default to 'current_session' if we can't determine
            current_condition = 'current_session'
            if len(conditions_data) == 1:
                current_condition = list(conditions_data.keys())[0]
            elif 'cooperation' in conditions_data:
                current_condition = 'cooperation'
            elif 'competition' in conditions_data:
                current_condition = 'competition'
                
            conditions_data[current_condition] = {
                'pair_01': {
                    'sync_results': sync_results,
                    'obj_cols': obj_cols
                }
            }
            
            print(f"[OK] Loaded current session data as condition: {current_condition}")
    
    return conditions_data

def calculate_condition_statistics(conditions_data):
    """
    Calculate synchronization statistics for each condition.
    """
    stats = {}
    
    for condition, pairs_data in conditions_data.items():
        if not pairs_data:
            continue
            
        pair_sync_percentages = []
        total_frames = 0
        
        for pair_id, pair_data in pairs_data.items():
            if 'sync_results' in pair_data:
                sync_results = pair_data['sync_results']
                
                # Calculate sync percentage for this pair
                has_sync = sum(1 for r in sync_results if r['sync_count'] > 0)
                total_pair_frames = len(sync_results)
                sync_percentage = (has_sync / total_pair_frames) * 100 if total_pair_frames > 0 else 0
                
                pair_sync_percentages.append(sync_percentage)
                total_frames += total_pair_frames
                
                print(f"[INFO] {condition}/{pair_id}: {sync_percentage:.1f}% sync ({has_sync}/{total_pair_frames} frames)")
        
        if pair_sync_percentages:
            stats[condition] = {
                'avg_sync_percent': np.mean(pair_sync_percentages),
                'std_sync_percent': np.std(pair_sync_percentages) if len(pair_sync_percentages) > 1 else 0,
                'pairs': list(pairs_data.keys()),
                'pair_percentages': pair_sync_percentages,
                'total_frames': total_frames,
                'num_pairs': len(pair_sync_percentages)
            }
    
    return stats

def create_histogram(stats, output_dir):
    """
    Create histogram showing average synchronization percentage by condition.
    """
    if not stats:
        print("[WARN] No statistics to plot")
        return None
        
    conditions = list(stats.keys())
    averages = [stats[cond]['avg_sync_percent'] for cond in conditions]
    std_devs = [stats[cond]['std_sync_percent'] for cond in conditions]
    
    # Color mapping
    colors = {
        'solo': '#FF6B6B',
        'cooperation': '#4ECDC4', 
        'competition': '#45B7D1',
        'current_session': '#9B59B6'
    }
    plot_colors = [colors.get(cond, '#333333') for cond in conditions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, averages, yerr=std_devs, capsize=5, 
                   color=plot_colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, avg, std in zip(bars, averages, std_devs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{avg:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Average Gaze Synchronization by Condition', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Synchronization Percentage (%)', fontsize=12)
    plt.xlabel('Condition', fontsize=12)
    plt.ylim(0, max(averages) + max(std_devs) + 10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add sample size annotations
    for i, (condition, avg) in enumerate(zip(conditions, averages)):
        n_pairs = stats[condition]['num_pairs']
        n_frames = stats[condition]['total_frames']
        plt.text(i, -5, f'n={n_pairs} pairs\n({n_frames} frames)', 
                ha='center', va='top', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    histogram_path = os.path.join(output_dir, 'synchronization_histogram.png')
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved histogram: {histogram_path}")
    
    # Save statistical values for further analysis
    stats_file = os.path.join(output_dir, 'synchronization_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved statistics: {stats_file}")
    
    return histogram_path

def calculate_timeline_probabilities(conditions_data):
    """
    Calculate synchronization probability over time for each condition.
    """
    timeline_probs = {}
    
    for condition, pairs_data in conditions_data.items():
        if not pairs_data:
            continue
            
        # Collect all sync results from all pairs in this condition
        all_sync_results = []
        for pair_data in pairs_data.values():
            if 'sync_results' in pair_data:
                all_sync_results.append(pair_data['sync_results'])
        
        if not all_sync_results:
            continue
            
        # Find the frame range across all pairs
        all_frames = set()
        for sync_results in all_sync_results:
            for result in sync_results:
                all_frames.add(result['frame'])
        
        if not all_frames:
            continue
            
        all_frames = sorted(list(all_frames))
        prob_by_frame = []
        
        # Calculate probability for each frame
        for frame_num in all_frames:
            sync_values = []
            
            # Collect sync status from all pairs for this frame
            for sync_results in all_sync_results:
                frame_result = None
                for result in sync_results:
                    if result['frame'] == frame_num:
                        frame_result = result
                        break
                
                if frame_result:
                    # 1 if any synchronization occurred, 0 otherwise
                    sync_value = 1 if frame_result['sync_count'] > 0 else 0
                    sync_values.append(sync_value)
            
            # Calculate probability as average across all pairs
            if sync_values:
                probability = np.mean(sync_values)
                prob_by_frame.append(probability)
            else:
                prob_by_frame.append(0)
        
        timeline_probs[condition] = {
            'frames': all_frames,
            'probabilities': prob_by_frame
        }
        
        print(f"[INFO] {condition}: Timeline with {len(all_frames)} frames, "
              f"avg probability: {np.mean(prob_by_frame):.3f}")
    
    return timeline_probs

def create_timeline_graph(timeline_probs, output_dir):
    """
    Create timeline graph showing synchronization probability over time.
    """
    if not timeline_probs:
        print("[WARN] No timeline data to plot")
        return None
        
    plt.figure(figsize=(14, 8))
    
    colors = {
        'solo': '#FF6B6B',
        'cooperation': '#4ECDC4', 
        'competition': '#45B7D1',
        'current_session': '#9B59B6'
    }
    
    for condition, data in timeline_probs.items():
        frames = data['frames']
        probs = data['probabilities']
        
        color = colors.get(condition, '#333333')
        plt.plot(frames, probs, label=condition.replace('_', ' ').title(), 
                linewidth=2, color=color, alpha=0.8, marker='o', markersize=3)
        
        # Add smoothed trend line if we have enough data points
        if len(probs) > 10:
            try:
                from scipy import ndimage
                smoothed = ndimage.gaussian_filter1d(probs, sigma=2)
                plt.plot(frames, smoothed, '--', color=color, alpha=0.5, linewidth=1)
            except ImportError:
                # Fallback to simple moving average if scipy not available
                window = min(5, len(probs) // 4)
                if window > 1:
                    smoothed = pd.Series(probs).rolling(window, center=True).mean()
                    plt.plot(frames, smoothed, '--', color=color, alpha=0.5, linewidth=1)
    
    plt.title('Synchronization Probability Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Point (Frame Number)', fontsize=12)
    plt.ylabel('Synchronization Probability', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add horizontal reference lines
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.text(max(frames) * 0.02, 0.52, '50% probability', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    timeline_path = os.path.join(output_dir, 'synchronization_timeline.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved timeline: {timeline_path}")
    
    # Save timeline data for further analysis
    timeline_data_file = os.path.join(output_dir, 'timeline_probabilities.json')
    with open(timeline_data_file, 'w') as f:
        json.dump(timeline_probs, f, indent=2)
    print(f"[OK] Saved timeline data: {timeline_data_file}")
    
    return timeline_path

def create_aoi_breakdown(conditions_data, output_dir):
    """
    Create individual AOI synchronization analysis for each condition and pair.
    """
    aoi_results = {}
    
    for condition, pairs_data in conditions_data.items():
        if not pairs_data:
            continue
            
        aoi_stats = {}
        
        # Get object columns from first available pair
        obj_cols = None
        for pair_data in pairs_data.values():
            if 'obj_cols' in pair_data:
                obj_cols = pair_data['obj_cols']
                break
        
        if not obj_cols:
            continue
            
        for obj_col in obj_cols:
            obj_name = obj_col.replace('obj_', 'Object ')
            pair_percentages = []
            pair_details = {}
            
            for pair_id, pair_data in pairs_data.items():
                if 'sync_results' not in pair_data:
                    continue
                    
                sync_results = pair_data['sync_results']
                
                # Calculate sync rate for this object in this pair
                obj_sync_count = 0
                total_frames = len(sync_results)
                
                for result in sync_results:
                    if obj_col in result.get('object_sync_status', {}):
                        if result['object_sync_status'][obj_col]:
                            obj_sync_count += 1
                
                sync_rate = (obj_sync_count / total_frames * 100) if total_frames > 0 else 0
                pair_percentages.append(sync_rate)
                pair_details[pair_id] = {
                    'sync_rate': sync_rate,
                    'sync_frames': obj_sync_count,
                    'total_frames': total_frames
                }
                
                print(f"[INFO] {condition}/{pair_id}/{obj_name}: {sync_rate:.1f}% "
                      f"({obj_sync_count}/{total_frames} frames)")
            
            if pair_percentages:
                aoi_stats[obj_name] = {
                    'avg_percent': np.mean(pair_percentages),
                    'std_percent': np.std(pair_percentages) if len(pair_percentages) > 1 else 0,
                    'pair_percentages': pair_percentages,
                    'pair_details': pair_details
                }
        
        if aoi_stats:
            aoi_results[condition] = aoi_stats
    
    # Create AOI breakdown visualization
    if not aoi_results:
        print("[WARN] No AOI data to plot")
        return None, aoi_results
        
    conditions = list(aoi_results.keys())
    aoi_names = list(next(iter(aoi_results.values())).keys())
    
    if not aoi_names:
        print("[WARN] No AOI names found")
        return None, aoi_results
    
    fig, axes = plt.subplots(len(aoi_names), 1, figsize=(12, 4 * len(aoi_names)))
    if len(aoi_names) == 1:
        axes = [axes]
    
    colors = {
        'solo': '#FF6B6B',
        'cooperation': '#4ECDC4', 
        'competition': '#45B7D1',
        'current_session': '#9B59B6'
    }
    
    for i, aoi in enumerate(aoi_names):
        ax = axes[i]
        
        # Get data for all conditions for this AOI
        aoi_conditions = []
        avg_percentages = []
        std_percentages = []
        plot_colors = []
        
        for condition in conditions:
            if aoi in aoi_results[condition]:
                aoi_conditions.append(condition)
                avg_percentages.append(aoi_results[condition][aoi]['avg_percent'])
                std_percentages.append(aoi_results[condition][aoi]['std_percent'])
                plot_colors.append(colors.get(condition, '#333333'))
        
        if not aoi_conditions:
            continue
            
        bars = ax.bar(aoi_conditions, avg_percentages, yerr=std_percentages, capsize=5,
                     color=plot_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, avg in zip(bars, avg_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{avg:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{aoi} - Synchronization by Condition', 
                    fontweight='bold', fontsize=14)
        ax.set_ylabel('Synchronization (%)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(avg_percentages) * 1.2 if avg_percentages else 1)
    
    plt.tight_layout()
    aoi_path = os.path.join(output_dir, 'aoi_breakdown.png')
    plt.savefig(aoi_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved AOI breakdown: {aoi_path}")
    
    # Save AOI data for further analysis
    aoi_data_file = os.path.join(output_dir, 'aoi_breakdown_data.json')
    with open(aoi_data_file, 'w') as f:
        json.dump(aoi_results, f, indent=2)
    print(f"[OK] Saved AOI data: {aoi_data_file}")
    
    return aoi_path, aoi_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python custom_multi_condition_analysis.py <results_directory> [workspace_directory]")
        print("\nThis script analyzes gaze synchronization data and creates:")
        print("1. Histogram of average synchronization by condition")
        print("2. Timeline showing synchronization probability over time")
        print("3. Individual AOI breakdown analysis")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    workspace_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=== Custom Multi-Condition Gaze Synchronization Analysis ===\n")
    
    # Create output directory
    output_dir = os.path.join(results_dir, "multi_condition_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Load data from all available sources
    print("Loading synchronization data...")
    conditions_data = load_all_conditions_data(results_dir, workspace_dir)
    
    if not conditions_data:
        print("Error: No synchronization data found")
        sys.exit(1)
    
    print(f"[INFO] Found {len(conditions_data)} conditions: {list(conditions_data.keys())}")
    
    # Calculate statistics
    print("\nCalculating condition statistics...")
    stats = calculate_condition_statistics(conditions_data)
    
    if not stats:
        print("Error: No statistics calculated")
        sys.exit(1)
    
    # Print summary
    print("\n=== CONDITION SUMMARY ===")
    for condition, stat in stats.items():
        print(f"{condition.upper()}:")
        print(f"  - Average synchronization: {stat['avg_sync_percent']:.1f}% ± {stat['std_sync_percent']:.1f}%")
        print(f"  - Number of pairs: {stat['num_pairs']}")
        print(f"  - Total frames: {stat['total_frames']}")
        print()
    
    # Create visualizations
    print("Creating histogram...")
    histogram_path = create_histogram(stats, output_dir)
    
    print("Calculating timeline probabilities...")
    timeline_probs = calculate_timeline_probabilities(conditions_data)
    
    print("Creating timeline graph...")
    timeline_path = create_timeline_graph(timeline_probs, output_dir)
    
    print("Creating AOI breakdown...")
    aoi_path, aoi_results = create_aoi_breakdown(conditions_data, output_dir)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"All results saved to: {output_dir}")
    print("Generated files:")
    if histogram_path:
        print(f"  - synchronization_histogram.png")
        print(f"  - synchronization_statistics.json")
    if timeline_path:
        print(f"  - synchronization_timeline.png") 
        print(f"  - timeline_probabilities.json")
    if aoi_path:
        print(f"  - aoi_breakdown.png")
        print(f"  - aoi_breakdown_data.json")
    
    print(f"\nFor statistical analysis, check the JSON files in {output_dir}")

if __name__ == "__main__":
    main()