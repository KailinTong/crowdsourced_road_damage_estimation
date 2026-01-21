
import os
import json
import subprocess
import glob

# Mappings
RISK_MAPPING = {
    "dark_red": 0.9,
    "light_red": 0.8,
    "dark_orange": 0.7,
    "orange": 0.6,
    "light_orange": 0.5,
    "light_yellow": 0.4,
    "pale_yellow": 0.3,
    "very_light_green": 0.1
}

INPUT_DIR = "data/Graz_A2-Highway_electronic-file-repository/risk-zones"
OUTPUT_JSON = "data/Graz_A2/road_anomaly_probabilities.json"
OUTPUT_XML = "data/Graz_A2/risk_zones.add.xml"

# Projection info from scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml
PROJ_PARAMS = "+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
NET_OFFSET_X = -531721.08
NET_OFFSET_Y = -5205448.90

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
        
        # Determine risk level
        risk_level = "very_light_green" # default
        for key in RISK_MAPPING:
            if key in filename:
                risk_level = key
                break
        
        probability = RISK_MAPPING.get(risk_level, 0.1)
        
        # Determine severity
        severity = 'l'
        if probability > 0.7:
            severity = 'h'
        elif probability > 0.4:
            severity = 'm'
            
        print(f"Processing {filename} with risk {risk_level} (prob: {probability})...")
        
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
                    sumo_coords_str_list = []
                    wkt_coords_str_list = []
                    
                    for pt in exterior_ring:
                        x_utm, y_utm = pt[0], pt[1]
                        x_sumo = x_utm + NET_OFFSET_X
                        y_sumo = y_utm + NET_OFFSET_Y
                        transformed_coords.append((x_sumo, y_sumo))
                        sumo_coords_str_list.append(f"{x_sumo:.2f},{y_sumo:.2f}")
                        wkt_coords_str_list.append(f"{x_sumo} {y_sumo}")
                    
                    # Create WKT manually
                    wkt_str = f"POLYGON (({', '.join(wkt_coords_str_list)}))"
                    
                    # ID
                    pothole_id = f"pothole_{pothole_counter}"
                    
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
                    
                    # XML Entry
                    shape_str = " ".join(sumo_coords_str_list)
                    xml_content += f'    <poly id="{pothole_id}" color="red" fill="1" layer="100.00" shape="{shape_str}"/>\n'
                    
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
