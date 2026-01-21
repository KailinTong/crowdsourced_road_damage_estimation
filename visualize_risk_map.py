
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
from shapely.wkt import loads as load_wkt

# Configuration
NET_FILE = "scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml"
PROB_FILE = "data/Graz_A2/road_anomaly_probabilities.json"
OUTPUT_IMG = "image/Graz_A2_risk_visualization.png"

def parse_lanes(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()
    lanes = []
    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            shape_str = lane.attrib["shape"]
            # shape="x1,y1 x2,y2 ..."
            pts = []
            for pair in shape_str.strip().split():
                coords = list(map(float, pair.split(",")))
                x, y = coords[0], coords[1]
                pts.append((x, y))
            lanes.append(pts)
    return lanes

def parse_risk_zones(prob_file):
    with open(prob_file, 'r') as f:
        data = json.load(f)
    
    zones = []
    for key, val in data.items():
        wkt = val.get("polygon")
        poly = load_wkt(wkt)
        
        # Determine color based on probability
        # Mapping used:
        # 0.9: dark_red
        # 0.8: light_red
        # 0.7: dark_orange
        # 0.6: orange
        # 0.5: light_orange
        # 0.4: light_yellow
        # 0.3: pale_yellow
        # 0.1: very_light_green (stable)
        
        prob = val.get("probabilities", {}).get("vehicle", 0.1)
        
        color = 'green' # default
        if prob >= 0.9: color = '#8B0000' # dark red
        elif prob >= 0.8: color = '#FF0000' # red
        elif prob >= 0.7: color = '#FF8C00' # dark orange
        elif prob >= 0.6: color = '#FFA500' # orange
        elif prob >= 0.5: color = '#FFCC00' # light orange (goldish)
        elif prob >= 0.4: color = '#FFFFE0' # light yellow
        elif prob >= 0.3: color = '#FFFACD' # pale yellow
        else: color = '#90EE90' # light green
        
        zones.append((poly, color, prob))
    return zones

def main():
    print(f"Loading network: {NET_FILE}")
    lanes = parse_lanes(NET_FILE)
    
    print(f"Loading risk zones: {PROB_FILE}")
    zones = parse_risk_zones(PROB_FILE)
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot Lanes
    for lane_pts in lanes:
        xs, ys = zip(*lane_pts)
        ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
        
    # Plot Zones
    for poly, color, prob in zones:
        # Shapely polygon to matplotlib patch
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=0.6, zorder=2, label=f"Prob {prob}")
        elif poly.geom_type == 'MultiPolygon':
            for subpoly in poly.geoms:
                x, y = subpoly.exterior.xy
                ax.fill(x, y, color=color, alpha=0.6, zorder=2)

    # Create custom legend
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
    
    ax.set_title("Graz A2: Risk Zones Overlaid on Road Network")
    ax.set_xlabel("X (UTM 33N Local)")
    ax.set_ylabel("Y (UTM 33N Local)")
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"Saved visualization to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
