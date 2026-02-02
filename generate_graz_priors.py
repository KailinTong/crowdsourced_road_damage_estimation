
import os
import json
import subprocess
import glob
import re
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union

# Prior and Severity Config
UNIFORM_PRIOR = 0.6  # Can be changed for multiple runs
YEARS = 5            # Projection period for cumulative deviation

# Mapping from shapefile color to "deviation_in_mm/year" based on doc/image/graz-a2-insar.png
DEVIATION_IN_MM_PER_YEAR = {
    "dark_red": 5.0,
    "light_red": 4.5,
    "dark_orange": 4.0,
    "orange": 3.5,
    "light_orange": 3.0,
    "light_yellow": 2.5,
    "pale_yellow": 2.0,
    "very_light_green": 1.0
}

INPUT_DIR = "data/Graz_A2-Highway_electronic-file-repository/risk-zones"
OUTPUT_JSON = "data/Graz_A2/road_anomaly_probabilities.json"
OUTPUT_XML = "data/Graz_A2/risk_zones.add.xml"

# Projection info from scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml
PROJ_PARAMS = "+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
NET_OFFSET_X = -531721.08
NET_OFFSET_Y = -5205448.90

# Network boundary (convBoundary from net.xml)
NET_MIN_X = 0.0
NET_MIN_Y = 0.0
NET_MAX_X = 4810.70
NET_MAX_Y = 1782.88

# Network file for road geometry
NET_FILE = "scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml"
ROAD_BUFFER = 2.0  # Buffer around road centerlines in meters (tight fit)

def get_road_geometry():
    """Extract road geometry from SUMO network and create buffered polygon."""
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    
    road_lines = []
    
    # Extract lane shapes (format: "x1,y1,z1 x2,y2,z2 ...")
    for lane in root.iter('lane'):
        shape_str = lane.get('shape')
        if not shape_str:
            continue
        
        points = []
        for pt in shape_str.split():
            coords = pt.split(',')
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                points.append((x, y))
        
        if len(points) >= 2:
            road_lines.append(LineString(points))
    
    # Buffer all lines and union them into a single polygon
    if road_lines:
        buffered = [line.buffer(ROAD_BUFFER) for line in road_lines]
        road_polygon = unary_union(buffered)
        print(f"Created road polygon from {len(road_lines)} lane segments")
        return road_polygon
    
    return None

def clip_coords_to_boundary(coords):
    """Clip coordinates to network boundary and return filtered list."""
    clipped = []
    for x, y in coords:
        # Clamp coordinates to boundary
        x_clamped = max(NET_MIN_X, min(NET_MAX_X, x))
        y_clamped = max(NET_MIN_Y, min(NET_MAX_Y, y))
        clipped.append((x_clamped, y_clamped))
    return clipped

def is_polygon_in_boundary(coords):
    """Check if at least part of polygon overlaps with network boundary."""
    for x, y in coords:
        if NET_MIN_X <= x <= NET_MAX_X and NET_MIN_Y <= y <= NET_MAX_Y:
            return True
    return False

def convert_shp_to_geojson_projected(shp_path):
    geojson_path = shp_path.replace(".shp", "_temp.geojson")
    try:
        # Reproject to UTM Zone 33N
        subprocess.run(
            ["ogr2ogr", "-f", "GeoJSON", "-t_srs", PROJ_PARAMS, geojson_path, shp_path],
            check=True, capture_output=True
        )
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        os.remove(geojson_path)
        return data
    except subprocess.CalledProcessError as e:
        print(f"Error converting {shp_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {geojson_path}: {e}")
        return None

def main():
    final_json = {}
    pothole_counter = 0
    
    # Get road geometry for clipping
    print("Extracting road geometry from network...")
    road_polygon = get_road_geometry()
    if road_polygon is None:
        print("Warning: Could not extract road geometry, using bounding box only")
    
    xml_header = """<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <!-- Shapes -->
"""
    xml_footer = "</additional>"
    xml_content = ""

    # Get all shapefiles
    shp_files = glob.glob(os.path.join(INPUT_DIR, "*.shp"))
    
    for shp_file in shp_files:
        filename = os.path.basename(shp_file).replace(".shp", "")
        
        # Determine annual rate and compute cumulative depth
        rate_mm_y = 1.0 # default
        for key in DEVIATION_IN_MM_PER_YEAR:
            if key in filename:
                rate_mm_y = DEVIATION_IN_MM_PER_YEAR[key]
                break
        
        depth_mm = rate_mm_y * YEARS
        
        # Determine severity based on strict ASTM D6433 thresholds
        # High: > 50mm, Medium: 25-50mm, Low: 13-25mm
        if depth_mm > 50:
            severity = 'h'
            color = 'red'
        elif depth_mm >= 25:
            severity = 'm'
            color = 'orange'
        elif depth_mm >= 13:
            severity = 'l'
            color = 'yellow'
        else:
            severity = 'mild_road'  # Too shallow to be an ASTM pothole
            color = 'blue'
        
        probability = UNIFORM_PRIOR
            
        print(f"Processing {filename}: {rate_mm_y}mm/y -> {depth_mm}mm ({YEARS}yrs), severity {severity}")
        
        geojson_data = convert_shp_to_geojson_projected(shp_file)
        
        if geojson_data and 'features' in geojson_data:
            for feature in geojson_data['features']:
                geometry = feature.get('geometry')
                if not geometry:
                    continue
                
                # Check geometry type
                geom_type = geometry.get('type')
                coords = geometry.get('coordinates')
                
                polygons = []
                if geom_type == 'Polygon':
                    polygons.append(coords)
                elif geom_type == 'MultiPolygon':
                    polygons.extend(coords)
                else:
                    continue
                
                for poly_coords in polygons:
                    # poly_coords is list of rings. First ring is exterior.
                    if not poly_coords:
                        continue
                    
                    exterior_ring = poly_coords[0] 
                    
                    # Transform coordinates
                    transformed_coords = []
                    
                    for pt in exterior_ring:
                        x_utm, y_utm = pt[0], pt[1]
                        x_sumo = x_utm + NET_OFFSET_X
                        y_sumo = y_utm + NET_OFFSET_Y
                        transformed_coords.append((x_sumo, y_sumo))
                    
                    # Skip polygons completely outside network boundary
                    if not is_polygon_in_boundary(transformed_coords):
                        continue
                    
                    # Create shapely polygon from coordinates
                    try:
                        damage_poly = Polygon(transformed_coords)
                        if not damage_poly.is_valid:
                            damage_poly = damage_poly.buffer(0)  # Fix invalid polygons
                    except Exception:
                        continue
                    
                    # Intersect with road geometry if available
                    if road_polygon is not None:
                        if not damage_poly.intersects(road_polygon):
                            continue  # Skip if no intersection with roads
                        
                        clipped_poly = damage_poly.intersection(road_polygon)
                        
                        # Handle different geometry types after intersection
                        if clipped_poly.is_empty:
                            continue
                        
                        # Get coordinates from clipped polygon
                        if hasattr(clipped_poly, 'exterior'):
                            clipped_coords = list(clipped_poly.exterior.coords)
                        elif hasattr(clipped_poly, 'geoms'):
                            # MultiPolygon - take largest
                            largest = max(clipped_poly.geoms, key=lambda g: g.area if hasattr(g, 'area') else 0)
                            if hasattr(largest, 'exterior'):
                                clipped_coords = list(largest.exterior.coords)
                            else:
                                continue
                        else:
                            continue
                    else:
                        # Fallback to bounding box clipping
                        clipped_coords = clip_coords_to_boundary(transformed_coords)
                    
                    # Generate coordinate strings from clipped coords
                    sumo_coords_str_list = [f"{x:.2f},{y:.2f}" for x, y in clipped_coords]
                    wkt_coords_str_list = [f"{x} {y}" for x, y in clipped_coords]
                    
                    # Create WKT manually
                    wkt_str = f"POLYGON (({', '.join(wkt_coords_str_list)}))"
                    
                    # ID
                    pothole_id = f"pothole_{pothole_counter}"
                    
                    if severity == 'mild_road':
                        continue

                    # JSON Entry
                    final_json[pothole_id] = {
                        "id": pothole_id,
                        "polygon": wkt_str,
                        "probabilities": {
                            "left": 0.0,
                            "right": probability, 
                            "vehicle": probability
                        },
                        "severity": severity
                    }
                    
                    # XML Entry with severity-based color
                    shape_str = " ".join(sumo_coords_str_list)
                    xml_content += f'    <poly id="{pothole_id}" color="{color}" fill="1" layer="100.00" shape="{shape_str}"/>\n'
                    
                    pothole_counter += 1

    # Write JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_json, f, indent=4)
    print(f"Generated {OUTPUT_JSON} with {len(final_json)} zones.")
    
    # Write XML
    with open(OUTPUT_XML, 'w') as f:
        f.write(xml_header + xml_content + xml_footer)
    print(f"Generated {OUTPUT_XML}.")

if __name__ == "__main__":
    main()
