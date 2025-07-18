import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



class OccupancyGrid:
    def __init__(self, net_file, resolution=1.0, prior=0.1, margin=5.0, overlap_steps=20, decay_rate=0.1, smoothing_sigma=1.0):
        self.decay_rate = decay_rate
        self.smoothing_sigma = smoothing_sigma
        self.resolution = resolution
        self.default_lane_width = 3.2
        self.prior = prior

        # Compute map bounds based on lane shape coordinates
        self.x_min, self.x_max, self.y_min, self.y_max = self._compute_bounds(net_file, margin)
        self.width = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / resolution))
        self.overlap_steps = overlap_steps

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
        overlap_steps = self.overlap_steps  # how many extra “steps” of overlap
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

    def update_cell(self, i, j, p_true, p_false, detected):
        """
        Update the log-odds of cell (i,j) based on a new binary measurement.

        :param i:         grid‐row index
        :param j:         grid‐col index
        :param p_true:    P(z=1 | D)  (true positive rate)
        :param p_false:   P(z=1 | ¬D) (false alarm rate)
        :param detected:  bool, True if we observed z=1, False if z=0
        """

        # check if cell is valid
        if np.isnan(self.log_odds[i, j]):
            return

        # choose the correct likelihoods for this measurement
        if detected:
            p_z = p_true  # P(z=1 | D)
            p_nz = p_false  # P(z=1 | ¬D)
        else:
            p_z = 1.0 - p_true  # P(z=0 | D)
            p_nz = 1.0 - p_false  # P(z=0 | ¬D)

        # increment in log-odds form:
        #   l_sensor = log [ P(z|D) / P(z|¬D) ]
        l_sensor = np.log(p_z / p_nz)



        # Bayes update in log-odds domain
        self.log_odds[i, j] += l_sensor

    def _apply_hits(self, hits: np.ndarray, sensor) -> None:
        """
        Apply one batch of hits to self.log_odds, including decay & smoothing.
        """
        # sensor log-odds increment per hit
        L_hit = np.log(sensor.p_true / sensor.p_false)

        valid = ~np.isnan(self.log_odds)
        # 1) add hits
        self.log_odds[valid] += hits[valid] * L_hit
        # 2) decay toward prior
        a = self.decay_rate
        self.log_odds[valid] = (1 - a) * self.log_odds[valid] + a * np.log(self.prior)
        # 3) smooth
        self.log_odds[valid] = gaussian_filter(
            self.log_odds[valid],
            sigma=self.smoothing_sigma
        )

    def batch_update(self, log_path: str, sensor, batch_size: int = 360):
        """
        Read detections from `log_path` in batches of `batch_size` lines,
        accumulate hits per grid‐cell, then apply updates incrementally.
        """
        # prepare a hits‐array for one batch
        batch_hits = np.zeros_like(self.log_odds, dtype=int)

        with open(log_path, 'r') as f:
            for i, line in enumerate(f, start=1):
                _, xs, ys, det = line.split()
                if det != 'True':
                    continue

                cell = self.world_to_grid(float(xs), float(ys))
                if cell:
                    batch_hits[cell] += 1

                # every batch_size lines, or at end, apply & reset
                if i % batch_size == 0:
                    self._apply_hits(batch_hits, sensor)
                    batch_hits.fill(0)

        # leftover hits after the final full batch
        if np.any(batch_hits):
            self._apply_hits(batch_hits, sensor)

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
        net_file="scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml",
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