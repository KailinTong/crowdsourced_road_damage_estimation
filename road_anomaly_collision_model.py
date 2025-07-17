import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString
import math
import json

# -- Utility: standard normal CDF
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

# -- Wheel‐based hit probability under X ~ N(0, σ²)
def wheel_hit_prob(a, b, d_i, t, sigma):
    """
    a, b: anomaly lateral span [a,b] relative to lane center (m)
    d_i: lateral offset of wheel center from lane center (m)
    t: tire contact width (m)
    sigma: std. dev. of lateral deviation (m)
    Returns P_i = Φ((b - d_i + t/2)/σ) - Φ((a - d_i - t/2)/σ)
    """
    lower = (a - d_i - t/2) / sigma
    upper = (b - d_i + t/2) / sigma
    return max(0.0, normal_cdf(upper) - normal_cdf(lower))

# -- Vehicle & wheel parameters
VEHICLE_WIDTH = 1.8    # m
TRACK_WIDTH   = 1.6    # m (distance between left & right wheel centers)
TIRE_WIDTH    = 0.2    # m (approx. contact patch width)
SIGMA         = 0.3    # m (lateral deviation std. dev.)

# compute left/right wheel offsets from lane center
d_left  = -TRACK_WIDTH / 2.0
d_right =  TRACK_WIDTH / 2.0

NET_FILE = "scenario/pothole_Brussel_rural/osm_withProjParam.net.xml"
Polygon_File = "scenario/pothole_Brussel_rural/potholes.add.xml"

# color to severity mapping
# < !-- High - severity(H) → red, thick, top layer -->
# <!-- Low-severity (L) → yellow, thin, bottom layer -->
# <!-- Medium-severity (M) → orange, medium, mid layer -->

color_to_severity = {"red": "H",
                     "yellow": "L",
                     "orange": "M"}


# -- Parse SUMO net for drivable lanes
net = ET.parse(NET_FILE).getroot()
lanes = []
for edge in net.findall("edge"):
    etype = edge.get("type")
    if not etype or etype == "highway.footway":
        continue
    for lane in edge.findall("lane"):
        shape = lane.get("shape")
        if not shape:
            continue
        width = float(lane.get("width")) if lane.get("width") else 3.2
        coords = [tuple(map(float, p.split(","))) for p in shape.split()]
        lanes.append({
            "geometry": LineString(coords),
            "width": width,
            "severity": color_to_severity.get(lane.get("color"), "H")
        })



# -- Parse potholes
add = ET.parse(Polygon_File).getroot()
potholes = []
for poly in add.findall("poly"):
    coords = [tuple(map(float, p.split(","))) for p in poly.get("shape").split()]
    potholes.append({
        "id": poly.get("id"),
        "polygon": Polygon(coords),
        "severtiy": color_to_severity.get(poly.get("color"), "H")
    })

# -- Compute per‐anomaly vehicle hit probability
results = dict()
for p in potholes:
    # find nearest lane
    lane = min(lanes, key=lambda L: p["polygon"].distance(L["geometry"]))
    # project anomaly centroid onto lane centerline
    proj = lane["geometry"].project(p["polygon"].centroid)
    base_pt = lane["geometry"].interpolate(proj)
    # approximate local normal
    eps = 1e-3
    p0 = lane["geometry"].interpolate(max(proj - eps, 0))
    p1 = lane["geometry"].interpolate(min(proj + eps, lane["geometry"].length))
    dx, dy = p1.x - p0.x, p1.y - p0.y
    mag = math.hypot(dx, dy) or 1e-6
    tvec = (dx/mag, dy/mag)
    n1 = (-tvec[1], tvec[0]); n2 = (tvec[1], -tvec[0])
    vc = (p["polygon"].centroid.x - base_pt.x, p["polygon"].centroid.y - base_pt.y)
    normal = n1 if (vc[0]*n1[0] + vc[1]*n1[1]) >= 0 else n2

    # compute anomaly lateral span [a, b]
    lats = []
    for x, y in p["polygon"].exterior.coords:
        rel = (x - base_pt.x, y - base_pt.y)
        lats.append(rel[0]*normal[0] + rel[1]*normal[1])
    a, b = min(lats), max(lats)

    # compute wheel probabilities
    P_l = wheel_hit_prob(a, b, d_left,  TIRE_WIDTH, SIGMA)
    P_r = wheel_hit_prob(a, b, d_right, TIRE_WIDTH, SIGMA)
    # combine for overall vehicle hit probability
    P_vehicle = 1.0 - (1.0 - P_l) * (1.0 - P_r)

    print(f"{p['id']}: P_left={P_l:.3f}, P_right={P_r:.3f}, P_vehicle={P_vehicle:.3f}, severity={p['severtiy']}")

    # severity is according to the color


    results[p["id"]] = {
        "id": p["id"],
        "polygon": p["polygon"].wkt,  # Save as WKT for easier JSON serialization
        "probabilities": { # only save three digits
            "left": round(P_l, 3),
            "right": round(P_r, 3),
            "vehicle": round(P_vehicle, 3)
        },
        "severity": p["severtiy"],  # severity is according to the color
    }
    #  save this dict to a json file
    with open('data/pothole_Brussel_rural/potholes_with_probabilities.json', 'w') as f:
        json.dump(results, f, indent=4)
#



