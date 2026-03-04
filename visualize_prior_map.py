#!/usr/bin/env python3
"""Visualize the prior probability map for Graz A2 scenario, reusing grid.py logic."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point
import xml.etree.ElementTree as ET

# Configuration - same as brussels_rural pattern
SCENARIO_NAME = "Graz_A2"
NET_FILE = "scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml"
PROBABILITY_FILE = "data/Graz_A2/road_anomaly_probabilities.json"
OUTPUT_IMG = f"image/{SCENARIO_NAME}/prior_probability_map.png"
RESOLUTION = 1.0
PRIOR_MILD = 0.05
MARGIN = 5.0

def get_network_bounds(net_file):
    """Extract network bounds from SUMO net file."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    location = root.find('.//location')
    if location is not None:
        conv_boundary = location.get('convBoundary')
        if conv_boundary:
            parts = conv_boundary.split(',')
            x_min, y_min, x_max, y_max = map(float, parts)
            return x_min - MARGIN, x_max + MARGIN, y_min - MARGIN, y_max + MARGIN
    return 0, 100, 0, 100

def load_risk_regions(prob_file):
    """Load risk regions from JSON file."""
    with open(prob_file, 'r') as f:
        prob_data = json.load(f)
    
    # Determine unique probability levels
    unique_probs = sorted(list(set(item['probabilities']['vehicle'] for item in prob_data.values())))
    prob_to_level = {p: i for i, p in enumerate(unique_probs)}
    risk_prior_by_level = {i: p for p, i in prob_to_level.items()}
    
    print(f"Loaded {len(prob_data)} risk zones")
    print(f"Risk levels: {risk_prior_by_level}")
    
    risk_regions = []
    for item in prob_data.values():
        poly = load_wkt(item['polygon'])
        prob = item['probabilities']['vehicle']
        level = prob_to_level[prob]
        risk_regions.append({'geometry': poly, 'level': level, 'prob': prob})
    
    return risk_regions, risk_prior_by_level

def build_prior_grid(x_min, x_max, y_min, y_max, resolution, risk_regions, risk_prior_by_level):
    """Build a 2D prior probability grid."""
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))
    
    # Initialize with baseline prior
    prior_grid = np.full((height, width), PRIOR_MILD)
    
    print(f"Grid size: {width} x {height}")
    
    for region in risk_regions:
        geom = region['geometry']
        prob = region['prob']
        
        minx, miny, maxx, maxy = geom.bounds
        i_min = max(int((miny - y_min) / resolution), 0)
        i_max = min(int(np.ceil((maxy - y_min) / resolution)), height)
        j_min = max(int((minx - x_min) / resolution), 0)
        j_max = min(int(np.ceil((maxx - x_min) / resolution)), width)
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                cx = x_min + j * resolution + resolution / 2
                cy = y_min + i * resolution + resolution / 2
                if geom.contains(Point(cx, cy)) or geom.touches(Point(cx, cy)):
                    prior_grid[i, j] = prob
    
    return prior_grid, width, height

def main():
    import os
    os.makedirs(f"image/{SCENARIO_NAME}", exist_ok=True)
    
    print(f"Loading network: {NET_FILE}")
    x_min, x_max, y_min, y_max = get_network_bounds(NET_FILE)
    print(f"Network bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
    
    print(f"Loading risk zones: {PROBABILITY_FILE}")
    risk_regions, risk_prior_by_level = load_risk_regions(PROBABILITY_FILE)
    
    print("Building prior grid...")
    prior_grid, width, height = build_prior_grid(x_min, x_max, y_min, y_max, RESOLUTION, 
                                                  risk_regions, risk_prior_by_level)
    
    print(f"Prior grid stats: min={prior_grid.min():.3f}, max={prior_grid.max():.3f}")
    
    # Plot using same color scheme as Brussels scenario (visualize_risk_map.py)
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create custom colormap matching Brussels scenario colors
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    
    # Define colors matching visualize_risk_map.py
    colors = [
        '#90EE90',  # 0.0-0.3: light green (stable)
        '#FFFACD',  # 0.3: pale yellow
        '#FFFFE0',  # 0.4: light yellow
        '#FFCC00',  # 0.5: light orange (goldish)
        '#FFA500',  # 0.6: orange
        '#FF8C00',  # 0.7: dark orange
        '#FF0000',  # 0.8: light red
        '#8B0000',  # 0.9: dark red
    ]
    bounds = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cmap = LinearSegmentedColormap.from_list('risk_cmap', colors, N=256)
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(prior_grid, origin='lower', extent=(x_min, x_max, y_min, y_max),
                   cmap=cmap, norm=norm, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, label='Prior Probability', ticks=bounds)
    
    # Create legend matching Brussels scenario
    legend_elements = [
        patches.Patch(facecolor='#8B0000', label='>0.9 (Dark Red)'),
        patches.Patch(facecolor='#FF0000', label='0.8 (Light Red)'),
        patches.Patch(facecolor='#FF8C00', label='0.7 (Dark Orange)'),
        patches.Patch(facecolor='#FFA500', label='0.6 (Orange)'),
        patches.Patch(facecolor='#FFCC00', label='0.5 (Light Orange)'),
        patches.Patch(facecolor='#FFFFE0', label='0.4 (Light Yellow)'),
        patches.Patch(facecolor='#FFFACD', label='0.3 (Pale Yellow)'),
        patches.Patch(facecolor='#90EE90', label='0.1 (Stable/Green)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Risk Probability")
    
    ax.set_title(f"{SCENARIO_NAME}: Prior Probability Map (InSAR-derived)")
    ax.set_xlabel("X (UTM 33N Local)")
    ax.set_ylabel("Y (UTM 33N Local)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"Saved visualization to {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()
