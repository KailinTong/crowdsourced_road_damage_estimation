from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET
import sumolib

class Damage:
    """
    Represents a single road-damage instance with a geometric shape.
    Can be initialized with a circular area or an arbitrary polygon.
    """
    def __init__(self, id: str = None, x: float = None, y: float = None,
                 radius: float = 1.0, shape: Polygon = None):
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

    def contains(self, x: float, y: float) -> bool:
        """
        Check if the point (x, y) falls within this damage's shape.
        """
        return self.shape.contains(Point(x, y))

    def __repr__(self) -> str:
        bounds = self.shape.bounds
        diameter = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        return (
            f"<Damage id={self.id!r} loc=({self.location.x:.2f},{self.location.y:.2f}) "
            f"diameter={diameter:.2f}>"
        )

class RoadDamage:
    """
    Load and store all damage instances (with shapes) from a SUMO net or an additional XML file.
    """
    def __init__(self, net_file: str, edge_ids: list[str],
                 radius: float = 1.0, damage_file: str = None):
        self.net_file = net_file
        self.edge_ids = edge_ids
        self.default_radius = radius
        if damage_file:
            # Read polygonal damages from SUMO additional file
            self._damages = self._read_from_xml(damage_file)
        else:
            # Generate circular damages at midpoints of specified edges
            self._damages = self._generate_damage_points()

    def _read_from_xml(self, xml_file: str) -> list[Damage]:
        """
        Parse a SUMO .add.xml file and create Damage instances for each <poly> entry.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        damages = []
        # Find all <poly> elements defining damaged polygons
        for poly in root.findall('poly'):
            pid = poly.get('id')
            shape_str = poly.get('shape')  # e.g. "x1,y1 x2,y2 ..."
            # Parse coordinate pairs
            coords = [tuple(map(float, pair.split(',')))
                      for pair in shape_str.split()]
            polygon = Polygon(coords)
            damages.append(Damage(id=pid, shape=polygon))
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
        Save the damage model to a file: writes one entry per line.
        For polygons, saves centroid coordinates; for points, saves the point.
        """
        with open(filename, 'w') as f:
            for damage in self._damages:
                x, y = damage.location.x, damage.location.y
                f.write(f"{damage.id} {x} {y}\n")

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
