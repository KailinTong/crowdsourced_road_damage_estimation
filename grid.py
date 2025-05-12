import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt



class OccupancyGrid:
    def __init__(self, net_file, resolution=1.0, prior=0.1, margin=5.0, oversample_margin=20):
        self.resolution = resolution
        self.default_lane_width = 3.2

        # Compute map bounds based on lane shape coordinates
        self.x_min, self.x_max, self.y_min, self.y_max = self._compute_bounds(net_file, margin)
        self.width = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / resolution))
        self.oversample_margin = oversample_margin

        self.log_odds = np.full((self.height, self.width), np.nan)
        self._parse_and_draw_lanes(net_file, p_z=prior)

    def _compute_bounds(self, net_file, margin):
        tree = ET.parse(net_file)
        root = tree.getroot()
        xs, ys = [], []

        for edge in root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                shape_str = lane.attrib["shape"]
                for pt in shape_str.strip().split():
                    x, y = map(float, pt.split(",")[:2])
                    xs.append(x)
                    ys.append(y)

        x_min = min(xs) - margin
        x_max = max(xs) + margin
        y_min = min(ys) - margin
        y_max = max(ys) + margin
        return x_min, x_max, y_min, y_max

    def visualize_lane_centerline(self, net_file):
        tree = ET.parse(net_file)
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                shape_str = lane.attrib["shape"]
                shape_pts = [tuple(map(float, pt.split(","))) for pt in shape_str.strip().split()]
                xs, ys = zip(*[(x, y) for x, y, *_ in shape_pts])
                plt.plot(xs, ys, linestyle='-', linewidth=0.5, label=f"Lane {lane.attrib['id']}")

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("Lane Centerlines")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("lane_centerlines.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()



    def _parse_and_draw_lanes(self, net_file, p_z=0.9):
        tree = ET.parse(net_file)
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                shape_pts = [tuple(map(float, pt.split(","))) for pt in lane.attrib["shape"].split()]
                width = float(lane.attrib.get("width", self.default_lane_width))

                for p0, p1 in zip(shape_pts[:-1], shape_pts[1:]):
                    self._draw_lane_segment(p0, p1, width, p_z)

    def _draw_lane_segment(self, p0, p1, width, p_z):
        x0, y0 = p0[:2]
        x1, y1 = p1[:2]
        dx, dy = x1 - x0, y1 - y0
        seg_len = np.hypot(dx, dy)
        if seg_len == 0:
            return

        # -- geometric setup --
        # unit normal
        nx, ny = -dy / seg_len, dx / seg_len
        half_w = width / 2.0

        # how many grid cells across the half‐width (rounded up)
        half_w_cells = int(np.ceil(half_w / self.resolution))

        # convert probability to log-odds once
        l_z = self._prob_to_logodds(p_z)

        # -- along-segment sampling setup --
        base_steps = int(np.ceil(seg_len / self.resolution))  # #steps to cover the length
        overlap_steps = self.oversample_margin  # how many extra “steps” of overlap
        # ensure at least one sample beyond each end
        n_along = max(base_steps + overlap_steps, overlap_steps + 1)

        # step size (in t-units) between samples if covering [0,1] in n_along points
        dt = 1.0 / (n_along - 1)
        eps = overlap_steps * dt

        # sample from t = -eps … 1+eps so that adjacent segments overlap by overlap_steps
        ts = np.linspace(-eps, 1 + eps, n_along)

        # -- paint each cross-section slice --
        for t in ts:
            cx = x0 + t * dx
            cy = y0 + t * dy

            center_idx = self.world_to_grid(cx, cy)
            if center_idx is None:
                continue
            i0, j0 = center_idx

            # fill all cells within half_w_cells of the slice center
            for di in range(-half_w_cells, half_w_cells + 1):
                for dj in range(-half_w_cells, half_w_cells + 1):
                    i, j = i0 + di, j0 + dj
                    if not (0 <= i < self.height and 0 <= j < self.width):
                        continue
                    if np.isnan(self.log_odds[i, j]):
                        self.log_odds[i, j] = l_z


    def world_to_grid(self, x, y):
        i = int((y - self.y_min) / self.resolution)
        j = int((x - self.x_min) / self.resolution)
        if 0 <= i < self.height and 0 <= j < self.width:
            return i, j
        return None

    def update_cell(self, i, j, p_z):
        l_z = self._prob_to_logodds(p_z)
        if np.isnan(self.log_odds[i, j]):
            self.log_odds[i, j] = 0
        self.log_odds[i, j] += l_z

    def get_probability_map(self):
        prob_map = np.full_like(self.log_odds, np.nan)
        mask = ~np.isnan(self.log_odds)
        prob_map[mask] = self._logodds_to_prob(self.log_odds[mask])
        return prob_map

    @staticmethod
    def _prob_to_logodds(p):
        return np.log(p / (1.0 - p))

    @staticmethod
    def _logodds_to_prob(l):
        return 1.0 - 1.0 / (1.0 + np.exp(l))



if __name__ == "__main__":



    # Create grid (bounds auto-computed)
    grid = OccupancyGrid(
        net_file="scenario/A2_GO2GW_v2_MM.net.xml",
        resolution=1.0,
        prior=0.1
    )

    # visualize lane centerlines
    grid.visualize_lane_centerline("scenario/A2_GO2GW_v2_MM.net.xml")

    # Get occupancy probability map
    prob_map = grid.get_probability_map()
    masked = np.ma.masked_invalid(prob_map)  # mask NaNs

    # Plot
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # NaN = transparent/white

    img = plt.imshow(
        masked,
        cmap=cmap,
        origin='lower',
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        interpolation='nearest'
    )

    plt.colorbar(img, label='Occupancy Probability')
    plt.title("Occupancy Grid Map (Auto-Bounded)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True, linestyle=':', linewidth=0.5)

    # Save image
    plt.savefig("auto_bounded_occupancy_grid.png", dpi=300, bbox_inches='tight')
    plt.show()