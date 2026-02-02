#!/usr/bin/env python3
"""
Visualize clustered prior probability map for Graz A2 scenario.
Reuses the pattern from brussels_rural with L/M/H severity coloring.
Shows: mild_road (blue), pothole_h (red), pothole_m (orange), pothole_l (yellow)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import json
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point
import xml.etree.ElementTree as ET

# Configuration
SCENARIO_NAME = "Graz_A2"
NET_FILE = "scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml"
PROBABILITY_FILE = "data/Graz_A2/road_anomaly_probabilities.json"
OUTPUT_IMG = f"image/{SCENARIO_NAME}/clustered_prior_probability_map.png"
RESOLUTION = 1.0
PRIOR_MILD = 0.05
PRIOR_UNIFORM = 0.6
MARGIN = 5.0

# Image size to match doc/image/graz-a2-insar.png (620x866)
OUTPUT_WIDTH = 620
OUTPUT_HEIGHT = 866

# Focus area (precisely aligned with doc/image/graz-a2-insar.png)
# InSAR image: 15°26.10'E to 15°27.00'E, 47°0.06'N to 47°1.14'N
# Converted to SUMO coordinates using UTM 33N projection + network offset
FOCUS_X_MIN = 1350
FOCUS_X_MAX = 2480
FOCUS_Y_MIN = 0       # Network starts at 0
FOCUS_Y_MAX = 1783    # Full network height (image covers full Y range)

# Colors matching brussels_rural pattern
COLORS = {
    'mild_road': (0.0, 0.0, 0.5),   # dark blue
    'pothole_h': (1.0, 0.0, 0.0),   # red (high severity)
    'pothole_m': (1.0, 0.5, 0.0),   # orange (medium severity)
    'pothole_l': (1.0, 1.0, 0.0),   # yellow (low severity)
}

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

def parse_lanes(net_file):
    """Extract lane geometries from network file."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    lanes = []
    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            shape_str = lane.attrib.get("shape", "")
            if not shape_str:
                continue
            pts = []
            for pair in shape_str.strip().split():
                coords = pair.split(",")
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    pts.append((x, y))
            if pts:
                lanes.append(pts)
    return lanes

def load_risk_regions(prob_file):
    """Load risk regions from JSON file."""
    with open(prob_file, 'r') as f:
        prob_data = json.load(f)
    
    regions = []
    for key, item in prob_data.items():
        poly = load_wkt(item['polygon'])
        prob = item['probabilities']['vehicle']
        severity = item.get('severity', 'm')  # default to medium
        
        # Determine type based on severity
        if severity == 'h':
            anomaly_type = 'pothole_h'
        elif severity == 'm':
            anomaly_type = 'pothole_m'
        else:
            anomaly_type = 'pothole_l'
        
        regions.append({
            'id': key,
            'geometry': poly,
            'prob': prob,
            'type': anomaly_type
        })
    
    print(f"Loaded {len(regions)} risk zones")
    
    # Count by type
    type_counts = {}
    for r in regions:
        type_counts[r['type']] = type_counts.get(r['type'], 0) + 1
    print(f"Type distribution: {type_counts}")
    
    return regions

def build_clustered_grid(x_min, x_max, y_min, y_max, resolution, lanes, risk_regions):
    """Build grid with mild_road and anomaly zones."""
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))
    
    # Initialize grids
    prob_grid = np.full((height, width), np.nan)
    type_grid = np.full((height, width), '', dtype=object)
    id_grid = np.full((height, width), 'na', dtype=object)
    
    print(f"Grid size: {width} x {height}")
    
    # Draw road lanes as mild_road
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    
    road_lines = []
    for lane_pts in lanes:
        if len(lane_pts) >= 2:
            road_lines.append(LineString(lane_pts))
    
    if road_lines:
        road_buffer = unary_union([line.buffer(5.0) for line in road_lines])  # 5m buffer
        
        # Draw road on grid
        minx, miny, maxx, maxy = road_buffer.bounds
        i_min = max(int((miny - y_min) / resolution), 0)
        i_max = min(int(np.ceil((maxy - y_min) / resolution)), height)
        j_min = max(int((minx - x_min) / resolution), 0)
        j_max = min(int(np.ceil((maxx - x_min) / resolution)), width)
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                cx = x_min + j * resolution + resolution / 2
                cy = y_min + i * resolution + resolution / 2
                if road_buffer.contains(Point(cx, cy)):
                    prob_grid[i, j] = PRIOR_MILD
                    type_grid[i, j] = 'mild_road'
    
    # Overlay risk zones (anomalies)
    for region in risk_regions:
        geom = region['geometry']
        prob = region['prob']
        anomaly_type = region['type']
        region_id = region['id']
        
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
                    prob_grid[i, j] = prob
                    type_grid[i, j] = anomaly_type
                    id_grid[i, j] = region_id
    
    return prob_grid, type_grid, id_grid

def visualize_clustered_map(prob_grid, type_grid, id_grid, output_path, x_min, x_max, y_min, y_max):
    """Visualize the clustered probability map."""
    H, W = prob_grid.shape
    
    # Build RGBA image
    rgba = np.zeros((H, W, 4), dtype=float)
    
    # Color each cell based on type
    for type_name, color in COLORS.items():
        mask = type_grid == type_name
        if not np.any(mask):
            continue
        rgba[mask, :3] = color
        
        # Set alpha based on type
        if type_name == 'mild_road':
            rgba[mask, 3] = 0.85
        else:
            # Alpha proportional to probability for anomalies
            probs = prob_grid[mask]
            alpha = 0.4 + 0.6 * probs  # Range from 0.4 to 1.0
            rgba[mask, 3] = alpha
    
    # Create figure with size matching doc/image/graz-a2-insar.png (620x866 pixels)
    dpi = 100
    fig_width = OUTPUT_WIDTH / dpi
    fig_height = OUTPUT_HEIGHT / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(rgba, origin='lower', interpolation='none', extent=(x_min, x_max, y_min, y_max))
    
    # Create legend
    legend_patches = [
        patches.Patch(color=COLORS['mild_road'], label='Road Network (p=0.05)'),
        patches.Patch(color=COLORS['pothole_h'], label='High Severity (>50mm dev., p=0.6)'),
        patches.Patch(color=COLORS['pothole_m'], label='Medium Severity (25-50mm dev., p=0.6)'),
        patches.Patch(color=COLORS['pothole_l'], label='Low Severity (13-25mm dev., p=0.6)'),
    ]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, title='Risk Level')
    
    ax.set_title(f'Clustered Probability Map (Uniform Prior p={PRIOR_UNIFORM})')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.show()
    plt.close()

def main():
    import os
    os.makedirs(f"image/{SCENARIO_NAME}", exist_ok=True)
    
    print(f"Loading network: {NET_FILE}")
    net_x_min, net_x_max, net_y_min, net_y_max = get_network_bounds(NET_FILE)
    print(f"Network bounds: X=[{net_x_min:.1f}, {net_x_max:.1f}], Y=[{net_y_min:.1f}, {net_y_max:.1f}]")
    
    # Use focus area to match InSAR image
    x_min, x_max = FOCUS_X_MIN, FOCUS_X_MAX
    y_min, y_max = FOCUS_Y_MIN, FOCUS_Y_MAX
    print(f"Focus area: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
    
    print("Parsing lanes...")
    lanes = parse_lanes(NET_FILE)
    print(f"Found {len(lanes)} lane segments")
    
    print(f"Loading risk zones: {PROBABILITY_FILE}")
    risk_regions = load_risk_regions(PROBABILITY_FILE)
    
    print("Building clustered grid...")
    prob_grid, type_grid, id_grid = build_clustered_grid(
        x_min, x_max, y_min, y_max, RESOLUTION, lanes, risk_regions
    )
    
    print(f"Grid stats: min={np.nanmin(prob_grid):.3f}, max={np.nanmax(prob_grid):.3f}")
    
    print("Visualizing...")
    visualize_clustered_map(prob_grid, type_grid, id_grid, OUTPUT_IMG, x_min, x_max, y_min, y_max)

if __name__ == "__main__":
    main()
