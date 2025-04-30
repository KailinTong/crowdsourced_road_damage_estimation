from collections import defaultdict
from shapely.geometry import Point
import sumolib

def load_damage_points_fix(netfile, edge_ids):
    """Use sumolib to read in the network and pick the midpoint of each edge."""
    net = sumolib.net.readNet(netfile)
    pts = []
    for eid in edge_ids:
        edge = net.getEdge(eid)
        shape = edge.getShape()  # list of (x,y) along the polyline
        if shape:
            mid = shape[len(shape)//2]
            pts.append((eid, *mid))
    return pts

class Damage:
    """
    Represents a single road-damage instance with a geometric shape.
    Assumption: the damage is a circular area around a point.
    """
    def __init__(self, edge_id: str, x: float, y: float, radius: float = 1.0):
        self.edge_id = edge_id
        self.location = Point(x, y)
        # shape is a circular buffer around the damage point
        self.shape = self.location.buffer(radius)

    def contains(self, x: float, y: float) -> bool:
        """
        Check if the point (x,y) falls within this damage's shape.
        """
        return self.shape.contains(Point(x, y))

    def __repr__(self) -> str:
        return (
            f"<Damage edge={self.edge_id!r} "
            f"loc=({self.location.x:.2f},{self.location.y:.2f}) "
            f"radius={self.shape.bounds[2] - self.shape.bounds[0]:.2f}>"
        )

class RoadDamage:
    """
    Load and store all damage instances (with shapes) from a SUMO net.
    """
    def __init__(self, net_file: str, edge_ids: list[str], radius: float = 1.0):
        self.net_file = net_file
        self.edge_ids = edge_ids
        self.default_radius = radius
        self._damages = self._load_damage_points()

    def _load_damage_points(self) -> list[Damage]:
        # load_damage_points returns a list of tuples (edge_id, x, y)
        raw = load_damage_points_fix(self.net_file, self.edge_ids)
        return [Damage(eid, x, y, radius=self.default_radius)
                for eid, x, y in raw]

    def get_edges(self) -> list[str]:
        """Unique edges that have damage."""
        return list({d.edge_id for d in self._damages})

    def get_damages(self, edge_id: str) -> list[Damage]:
        """All Damage objects for a given edge."""
        return [d for d in self._damages if d.edge_id == edge_id]

    def all_damages(self) -> list[Damage]:
        """Flat list of all Damage objects."""
        return list(self._damages)

    def __repr__(self) -> str:
        return f"<RoadDamage edges={len(self.get_edges())} points={len(self._damages)}>"