#!/usr/bin/env python3
"""Multi-condition gaze synchronization analysis.

Analyzes synchronization across multiple conditions (solo, cooperation, competition)
and generates histogram and timeline visualizations.

Usage:
    python multi_condition_analysis.py <results_directory>
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

def load_condition_data(results_dir):
    """Load synchronization data for all conditions and pairs."""
    conditions_data = {}
    
    condition_dirs = [d for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d))]
    
    for condition in condition_dirs:
        condition_path = os.path.join(results_dir, condition)
        conditions_data[condition] = {}
        
        pair_dirs = [d for d in os.listdir(condition_path) 
                    if os.path.isdir(os.path.join(condition_path, d))]
        
        for pair in pair_dirs:
            sync_file = os.path.join(condition_path, pair, "sync_analysis.csv")
            if os.path.exists(sync_file):
                try:
                    sync_df = pd.read_csv(sync_file)
                    conditions_data[condition][pair] = sync_df
                    print(f"[OK] Loaded {condition}/{pair}: {len(sync_df)} frames")
                except Exception as e:
                    print(f"[WARN] Failed to load {sync_file}: {e}")
            else:
                print(f"[WARN] Missing sync_analysis.csv in {condition}/{pair}")
    
    return conditions_data

def calculate_condition_statistics(conditions_data):
    """Calculate synchronization statistics for each condition."""
    stats = {}
    
    for condition, pairs_data in conditions_data.items():
        if not pairs_data:
            continue
            
        pair_sync_percentages = []
        total_frames = 0
        
        for pair, sync_df in pairs_data.items():
            has_sync = (sync_df['sync_count'] > 0).sum()
            total_pair_frames = len(sync_df)
            sync_percentage = (has_sync / total_pair_frames) * 100 if total_pair_frames > 0 else 0
            
            pair_sync_percentages.append(sync_percentage)
            total_frames += total_pair_frames
        
        stats[condition] = {
            'avg_sync_percent': np.mean(pair_sync_percentages) if pair_sync_percentages else 0,
            'std_sync_percent': np.std(pair_sync_percentages) if pair_sync_percentages else 0,
            'pairs': list(pairs_data.keys()),
            'pair_percentages': pair_sync_percentages,
            'total_frames': total_frames,
            'num_pairs': len(pair_sync_percentages)
        }
    
    return stats

def create_histogram(stats, output_dir):
    """Create histogram showing average synchronization percentage by condition."""
    conditions = list(stats.keys())
    averages = [stats[cond]['avg_sync_percent'] for cond in conditions]
    std_devs = [stats[cond]['std_sync_percent'] for cond in conditions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, averages, yerr=std_devs, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    
    for bar, avg, std in zip(bars, averages, std_devs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{avg:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Gaze Synchronization by Condition', fontsize=16, fontweight='bold')
    plt.ylabel('Synchronization Percentage (%)', fontsize=12)
    plt.xlabel('Condition', fontsize=12)
    plt.ylim(0, max(averages) + max(std_devs) + 10)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (condition, avg) in enumerate(zip(conditions, averages)):
        n_pairs = stats[condition]['num_pairs']
        plt.text(i, -5, f'n={n_pairs} pairs', ha='center', va='top', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    histogram_path = os.path.join(output_dir, 'synchronization_histogram.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved histogram: {histogram_path}")
    return histogram_path

def calculate_timeline_probabilities(conditions_data):
    """Calculate synchronization probability over time for each condition."""
    timeline_probs = {}
    
    for condition, pairs_data in conditions_data.items():
        if not pairs_data:
            continue
            
        max_frame = 0
        for sync_df in pairs_data.values():
            max_frame = max(max_frame, sync_df['frame'].max())
        
        prob_by_frame = []
        
        for frame_num in range(1, max_frame + 1):
            frame_syncs = []
            
            for pair, sync_df in pairs_data.items():
                frame_data = sync_df[sync_df['frame'] == frame_num]
                if not frame_data.empty:
                    has_sync = 1 if frame_data.iloc[0]['sync_count'] > 0 else 0
                    frame_syncs.append(has_sync)
            
            if frame_syncs:
                probability = np.mean(frame_syncs)
                prob_by_frame.append(probability)
            else:
                prob_by_frame.append(0)
        
        timeline_probs[condition] = {
            'frames': list(range(1, max_frame + 1)),
            'probabilities': prob_by_frame
        }
    
    return timeline_probs

def create_timeline_graph(timeline_probs, output_dir):
    """Create timeline graph showing synchronization probability over time."""
    plt.figure(figsize=(14, 8))
    
    colors = {'solo': '#FF6B6B', 'cooperation': '#4ECDC4', 'competition': '#45B7D1'}
    
    for condition, data in timeline_probs.items():
        frames = data['frames']
        probs = data['probabilities']
        
        color = colors.get(condition, '#333333')
        plt.plot(frames, probs, label=condition.title(), linewidth=2, 
                color=color, alpha=0.8)
        
        if len(probs) > 10:
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter1d(probs, sigma=2)
            plt.plot(frames, smoothed, '--', color=color, alpha=0.5, linewidth=1)
    
    plt.title('Synchronization Probability Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time Point (Frame Number)', fontsize=12)
    plt.ylabel('Synchronization Probability', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% probability')
    
    plt.tight_layout()
    timeline_path = os.path.join(output_dir, 'synchronization_timeline.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved timeline: {timeline_path}")
    return timeline_path

def create_aoi_breakdown(conditions_data, output_dir):
    """Create individual AOI synchronization analysis for each condition."""
    aoi_results = {}
    
    for condition, pairs_data in conditions_data.items():
        if not pairs_data:
            continue
            
        sample_df = next(iter(pairs_data.values()))
        aoi_cols = [col for col in sample_df.columns if col.endswith('_sync')]
        
        aoi_stats = {}
        for aoi in aoi_cols:
            aoi_name = aoi.replace('_sync', '')
            pair_percentages = []
            
            for pair, sync_df in pairs_data.items():
                if aoi in sync_df.columns:
                    sync_rate = sync_df[aoi].mean()
                    pair_percentages.append(sync_rate * 100)
            
            aoi_stats[aoi_name] = {
                'avg_percent': np.mean(pair_percentages) if pair_percentages else 0,
                'std_percent': np.std(pair_percentages) if pair_percentages else 0,
                'pair_percentages': pair_percentages
            }
        
        aoi_results[condition] = aoi_stats
    
    # Create AOI breakdown visualization
    if aoi_results:
        conditions = list(aoi_results.keys())
        aoi_names = list(next(iter(aoi_results.values())).keys())
        
        fig, axes = plt.subplots(len(aoi_names), 1, figsize=(12, 4 * len(aoi_names)))
        if len(aoi_names) == 1:
            axes = [axes]
        
        for i, aoi in enumerate(aoi_names):
            ax = axes[i]
            
            avg_percentages = [aoi_results[cond][aoi]['avg_percent'] for cond in conditions]
            std_percentages = [aoi_results[cond][aoi]['std_percent'] for cond in conditions]
            
            bars = ax.bar(conditions, avg_percentages, yerr=std_percentages, capsize=5,
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
            
            for bar, avg in zip(bars, avg_percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{avg:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{aoi.replace("_", " ").title()} - Synchronization by Condition', 
                        fontweight='bold')
            ax.set_ylabel('Synchronization (%)')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        aoi_path = os.path.join(output_dir, 'aoi_breakdown.png')
        plt.savefig(aoi_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved AOI breakdown: {aoi_path}")
        return aoi_path, aoi_results
    
    return None, aoi_results

def save_statistical_data(stats, timeline_probs, aoi_results, output_dir):
    """Save statistical data for further analysis."""
    statistical_output = {
        'condition_statistics': stats,
        'timeline_probabilities': timeline_probs,
        'aoi_breakdown': aoi_results,
        'summary': {
            'conditions_analyzed': list(stats.keys()),
            'total_pairs': sum(s['num_pairs'] for s in stats.values()),
            'total_frames': sum(s['total_frames'] for s in stats.values())
        }
    }
    
    json_path = os.path.join(output_dir, 'statistical_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(statistical_output, f, indent=2)
    
    csv_data = []
    for condition, stat in stats.items():
        for i, pair_pct in enumerate(stat['pair_percentages']):
            csv_data.append({
                'condition': condition,
                'pair': stat['pairs'][i],
                'sync_percentage': pair_pct,
                'total_frames': stat['total_frames'] // stat['num_pairs']
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'condition_statistics.csv')
    csv_df.to_csv(csv_path, index=False)
    
    print(f"[OK] Saved statistical data: {json_path} and {csv_path}")
    return json_path, csv_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python multi_condition_analysis.py <results_directory>")
        print("\nExpected directory structure:")
        print("  results/")
        print("  ├── condition1/")
        print("  │   ├── pair_01/sync_analysis.csv")
        print("  │   └── pair_02/sync_analysis.csv")
        print("  └── condition2/")
        print("      └── pair_01/sync_analysis.csv")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found")
        sys.exit(1)
    
    print("=== Multi-Condition Gaze Synchronization Analysis ===\n")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../data/processed/figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading synchronization data...")
    conditions_data = load_condition_data(results_dir)
    
    if not conditions_data:
        print("Error: No synchronization data found")
        sys.exit(1)
    
    # Calculate statistics
    print("\nCalculating condition statistics...")
    stats = calculate_condition_statistics(conditions_data)
    
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
    create_histogram(stats, output_dir)
    
    print("Calculating timeline probabilities...")
    timeline_probs = calculate_timeline_probabilities(conditions_data)
    
    print("Creating timeline graph...")
    create_timeline_graph(timeline_probs, output_dir)
    
    print("Creating AOI breakdown...")
    aoi_path, aoi_results = create_aoi_breakdown(conditions_data, output_dir)
    
    print("Saving statistical data...")
    save_statistical_data(stats, timeline_probs, aoi_results, output_dir)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"All results saved to: {output_dir}")
    print("Generated files:")
    print("  - synchronization_histogram.png")
    print("  - synchronization_timeline.png")
    if aoi_path:
        print("  - aoi_breakdown.png")
    print("  - statistical_analysis.json")
    print("  - condition_statistics.csv")

if __name__ == "__main__":
    main()