import logging

from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET
import sumolib
import json
class Damage:
    """
    Represents a single road-damage instance with a geometric shape.
    Can be initialized with a circular area or an arbitrary polygon.
    """
    def __init__(self, id: str = None, x: float = None, y: float = None,
                 radius: float = 1.0, shape: Polygon = None, probability: float = 1.0, severity: str = "l"):
        self.id = id
        if shape is not None:
            # Use provided polygon shape
            self.shape = shape
            # Use centroid for location reference
            self.location = shape.centroid
        else:
            # Default to circular buffer around point
            self.location = Point(x, y)
            self.shape = self.location.buffer(radius)
        self.probability = probability
        self.severity = severity

    def contains(self, x: float, y: float) -> bool:
        """
        Check if the point (x, y) falls within this damage's shape.
        """
        return self.shape.contains(Point(x, y))

    def __repr__(self) -> str:
        bounds = self.shape.bounds
        diameter = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        return (f"<Damage id={self.id!r} loc=({self.location.x:.2f},{self.location.y:.2f}) diameter={diameter:.2f} "
                f" prob={self.probability}  severity={self.severity}>")

    def intersects_travel_path(self, last_x, last_y, x, y):
        pass


class RoadDamage:
    """
    Load and store all damage instances (with shapes) from a SUMO net or an additional XML file.
    """
    def __init__(self, net_file: str, edge_ids: list[str],
                 radius: float = 1.0, damage_file: str = None, probability_file: str = None):
        self.net_file = net_file
        self.edge_ids = edge_ids
        self.default_radius = radius
        if damage_file and probability_file:
            # Read polygonal damages from SUMO additional file
            self._damages = self._read_road_damge_files(damage_file, probability_file)
        else:
            # Generate circular damages at midpoints of specified edges
            self._damages = self._generate_damage_points()

    def _read_road_damge_files(self, xml_file: str, probability_file: str) -> list[Damage]:
        """
        Parse a SUMO .add.xml file and create Damage instances for each <poly> entry.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        damages = []

        # open the probability file (json format)
        with open(probability_file, 'r') as f:
            probability_file = json.load(f)


        # Find all <poly> elements defining damaged polygons
        for poly in root.findall('poly'):
            pid = poly.get('id')
            shape_str = poly.get('shape')  # e.g. "x1,y1 x2,y2 ..."
            # Parse coordinate pairs
            coords = [tuple(map(float, pair.split(',')))
                      for pair in shape_str.split()]
            polygon = Polygon(coords)

            # Get the probability for this polygon ID
            probability = probability_file.get(pid).get('probabilities').get('vehicle', None)
            # give a warning if no probability is found
            if probability is None:
                logging.WARNING(f'No probability found for polygon ID {pid} in {probability_file}. ')
            severity = probability_file.get(pid).get('severity', None)
            # give a warning if no severity is found
            if severity is None:
                logging.WARNING(f'No severity found for polygon ID {pid} in {probability_file}. ')

            damages.append(Damage(id=pid, shape=polygon, probability=probability, severity=severity))
        return damages

    def _generate_damage_points(self) -> list[Damage]:
        from collections import defaultdict
        from shapely.geometry import Point

        def load_damage_points(netfile, edge_ids):
            net = sumolib.net.readNet(netfile)
            pts = []
            for eid in edge_ids:
                edge = net.getEdge(eid)
                shape = edge.getShape()
                if shape:
                    mid = shape[len(shape) // 2]
                    pts.append((eid, *mid))
            return pts

        raw = load_damage_points(self.net_file, self.edge_ids)
        return [Damage(id=eid, x=x, y=y, radius=self.default_radius)
                for eid, x, y in raw]

    def get_edges(self) -> list[str]:
        """Unique edges (or polygon IDs) that have damage."""
        return list({d.id for d in self._damages})

    def get_damages(self, identifier: str) -> list[Damage]:
        """All Damage objects for a given edge ID or polygon ID."""
        return [d for d in self._damages if d.id == identifier]

    def all_damages(self) -> list[Damage]:
        """Flat list of all Damage objects."""
        return list(self._damages)

    def __repr__(self) -> str:
        return f"<RoadDamage damages={len(self._damages)}>"

    def save(self, filename: str):
        """
        Save the damage model to a json:
        For polygons, saves centroid coordinates; for points, saves the point.
        """
        data = []
        for damage in self._damages:
            if damage.shape.is_empty:
                continue
            if isinstance(damage.shape, Polygon):
                # Save polygon centroid
                data.append({
                    'id': damage.id,
                    'centroid': (damage.shape.centroid.x, damage.shape.centroid.y),
                    'probability': damage.probability
                })
            else:
                # Save point coordinates
                data.append({
                    'id': damage.id,
                    'point': (damage.location.x, damage.location.y),
                    'probability': damage.probability
                })
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    @staticmethod
    def read(filename: str) -> list[Damage]:
        """Read the damage model from a file."""
        damages = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    id, x, y = parts
                    damages.append(Damage(id=id, x=float(x), y=float(y)))
        return damages
