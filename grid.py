import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from utilities import load_road_anomaly_metrics
from road_anomaly_collision_model import NET_FILE
from scipy.ndimage import label, measurements
from matplotlib import cm
from sensor import VehicleSensor

class OccupancyGrid:
    def __init__(self, net_file, sensor, resolution=1.0, prior=0.8, margin=5.0, overlap_steps=20, decay_rate=0.1, smoothing_sigma=1.0, pro_threshold = 0.5):

        self.decay_rate = decay_rate
        self.smoothing_sigma = smoothing_sigma
        self.resolution = resolution
        self.default_lane_width = 3.2
        self.prior = prior # TODO: this will be updated based on each grid according to the satellite data
        # self.prob_dict = prob_dict
        self.pro_threshold = pro_threshold  # threshold for filtering results
        self.sensor = sensor
        self.prob_map_dict = None
        # remove "mild_road" from the probability dictionary if it exists and create a new dictionary
        # road_anomaly_prob_dict = { k: v for k, v in prob_dict.items() if k != "mild_road"} if prob_dict else None
        # self.road_anomaly_prob_dict = road_anomaly_prob_dict
        # self.road_anomaly_types = list(road_anomaly_prob_dict.keys()) if road_anomaly_prob_dict else []

        # Compute map bounds based on lane shape coordinates
        self.x_min, self.x_max, self.y_min, self.y_max = self._compute_bounds(net_file, margin)
        self.width = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / resolution))
        self.overlap_steps = overlap_steps

        # Initialize the log-odds dictionary, the key is the road anomaly types. and it will hold the log-odds values for each grid cell, the key
        self.log_odds_dict = {anomaly_type: np.full((self.height, self.width), np.nan) for anomaly_type in self.sensor.anomaly_types}
        # self.log_odds = np.full((self.height, self.width), np.nan)
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



    def _parse_and_draw_lanes(self, net_file, p_z):
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
                    for anomaly_type in self.sensor.anomaly_types:
                        # update the log-odds for this cell
                        if np.isnan(self.log_odds_dict[anomaly_type][i, j]):
                            self.log_odds_dict[anomaly_type][i, j] = l_z
                        # else:
                        #     self.log_odds_dict[anomaly_type][i, j] += l_z



    def world_to_grid(self, x, y):
        i = int((y - self.y_min) / self.resolution)
        j = int((x - self.x_min) / self.resolution)
        if 0 <= i < self.height and 0 <= j < self.width:
            return i, j
        return None

    # def update_cell(self, i, j, p_true, p_false, detected):
    #     """
    #     Update the log-odds of cell (i,j) based on a new binary measurement.
    #
    #     :param i:         grid‐row index
    #     :param j:         grid‐col index
    #     :param p_true:    P(z=1 | D)  (true positive rate)
    #     :param p_false:   P(z=1 | ¬D) (false alarm rate)
    #     :param detected:  bool, True if we observed z=1, False if z=0
    #     """
    #
    #     # check if cell is valid
    #     if np.isnan(self.log_odds[i, j]):
    #         return
    #
    #     # choose the correct likelihoods for this measurement
    #     if detected:
    #         p_z = p_true  # P(z=1 | D)
    #         p_nz = p_false  # P(z=1 | ¬D)
    #     else:
    #         p_z = 1.0 - p_true  # P(z=0 | D)
    #         p_nz = 1.0 - p_false  # P(z=0 | ¬D)
    #
    #     # increment in log-odds form:
    #     #   l_sensor = log [ P(z|D) / P(z|¬D) ]
    #     l_sensor = np.log(p_z / p_nz)
    #
    #
    #
    #     # Bayes update in log-odds domain
    #     self.log_odds[i, j] += l_sensor
    #
    # def _apply_hits(self, hits: np.ndarray, sensor) -> None:
    #     """
    #     Apply one batch of hits to self.log_odds, including decay & smoothing.
    #     """
    #     # sensor log-odds increment per hit
    #
    #     # Extend this to support sensor model with prob_dict for different road anomaly types
    #     # prob_dict is a dictionary with keys as road anomaly types and values as dictionaries with 'tp' , 'tn' and 'fü' probabilities
    #     L_hit = np.log(sensor.p_true / sensor.p_false)
    #
    #     valid = ~np.isnan(self.log_odds)
    #     # 1) add hits
    #     self.log_odds[valid] += hits[valid] * L_hit
    #     # 2) decay toward prior
    #     a = self.decay_rate
    #     self.log_odds[valid] = (1 - a) * self.log_odds[valid] + a * np.log(self.prior)
    #     # 3) smooth
    #     self.log_odds[valid] = gaussian_filter(
    #         self.log_odds[valid],
    #         sigma=self.smoothing_sigma
    #     )

    # TODO think about where to use the prob_dict, it has been repeated ...

    def _apply_hits_k_road_anomaly(self, hits_dict: dict) -> None:
        """
        Apply one batch of hits (per anomaly type) to self.log_odds, including decay & smoothing.

        Parameters:
            hits_dict (dict): keys are anomaly types, values are binary arrays of hits.

        """
        # valid = ~np.isnan(self.log_odds)
        valid_dict = {anomaly_type: ~np.isnan(self.log_odds_dict[anomaly_type]) for anomaly_type in self.sensor.anomaly_types}


        # Initialize an increment array
        # L_total = np.zeros_like(self.log_odds)
        L_total_dict = {anomaly_type: np.zeros_like(self.log_odds_dict[anomaly_type]) for anomaly_type in self.sensor.anomaly_types}

        # Loop through each anomaly type to accumulate increments
        for anomaly_type, hits in hits_dict.items():
            probs = self.sensor.prob_dict[anomaly_type]

            # Compute log-odds increment per anomaly type
            # L_hit = np.log(probs['tp'] / probs['fa'])
            L_hit = np.log(self.sensor.prob_dict[anomaly_type]['tp'] / self.sensor.prob_dict[anomaly_type]['fp'])

            # Accumulate increments weighted by hits
            valid = valid_dict[anomaly_type]
            L_total_dict[anomaly_type][valid] += hits[valid] * L_hit

            # 1) Apply accumulated increments
            self.log_odds_dict[anomaly_type][valid] += L_total_dict[anomaly_type][valid]

            # 2) Apply decay towards prior
            a = self.decay_rate
            prior_log_odds = np.log(self.prior / (1 - self.prior))
            self.log_odds_dict[anomaly_type][valid] = (1 - a) * self.log_odds_dict[anomaly_type][valid] + a * prior_log_odds

            # 3) Apply Gaussian smoothing
            self.log_odds_dict[anomaly_type][valid] = gaussian_filter(
            self.log_odds_dict[anomaly_type][valid],
            sigma=self.smoothing_sigma)

    def batch_update(self, log_path: str, batch_size: int = 360):
        """
        Read detections from `log_path` in batches of `batch_size` lines,
        accumulate hits per grid‐cell, then apply updates incrementally.


        """
        # hits_dict (dict): keys are anomaly types, values are binary arrays of hits.
        hits_dict = {}
        for anomaly_type in self.sensor.anomaly_types:
            hits_dict[anomaly_type] = np.zeros((self.height, self.width), dtype=int)

        with open(log_path, 'r') as f:
            for i, line in enumerate(f, start=1):
                _, xs, ys, det, road_anomaly_type = line.split()
                if det != 'True':
                    continue

                cell = self.world_to_grid(float(xs), float(ys))
                if cell:
                    # Increment the hit count for the specific anomaly type
                    if road_anomaly_type in hits_dict:
                        hits_dict[road_anomaly_type][cell] += 1

                # every batch_size lines, or at end, apply & reset
                if i % batch_size == 0:
                    # self._apply_hits(batch_hits, sensor)
                    self._apply_hits_k_road_anomaly(hits_dict)
                    # reset hits for next batch
                    hits_dict = {anomaly_type: np.zeros((self.height, self.width), dtype=int) for anomaly_type in self.sensor.anomaly_types}

        # leftover hits after the final full batch
        if any(np.any(hits) for hits in hits_dict.values()):
            self._apply_hits_k_road_anomaly(hits_dict)

    def gen_probability_map(self):
        prob_map_dict = {}
        for anomaly_type, log_odds in self.log_odds_dict.items():
            prob_map = np.full_like(self.log_odds_dict[anomaly_type], np.nan)
            mask = ~np.isnan(self.log_odds_dict[anomaly_type])
            prob_map[mask] = self._logodds_to_prob(self.log_odds_dict[anomaly_type][mask])
            prob_map_dict[anomaly_type] = self._logodds_to_prob(log_odds)

        self.prob_map_dict = prob_map_dict
        return prob_map_dict

    def filter_results(self, average_neighbors=False):
        """
        Generate a filtered map:
        - For each cell, determine the anomaly type with max probability.
        - Filter out those below threshold, set to fallback value.
        - Assign unique IDs to connected regions of the same anomaly type.
        - Optionally, average probabilities over each region.

        Returns:
            filtered_prob_map (np.ndarray): thresholded max probability map
            max_prob_map_type (np.ndarray): map of anomaly type string for max probability
            region_id_map (np.ndarray): map of region IDs (0 = no region, >0 = unique ID)
            avg_prob_map (np.ndarray or None): map with region-averaged probability (if average_neighbors)
        """

        if not self.road_anomaly_types:
            raise RuntimeError("No road anomaly types in OccupancyGrid.")

        # Step 1: Find per-cell max probability and type
        max_prob_map = np.full_like(self.log_odds_dict[self.road_anomaly_types[0]], np.nan)
        max_prob_map_type = np.full(max_prob_map.shape, '', dtype='U32')  # Unicode for type names

        for anomaly_type in self.road_anomaly_types:
            prob_map = self._logodds_to_prob(self.log_odds_dict[anomaly_type])
            mask = ~np.isnan(prob_map)

            # Where this anomaly has higher probability, update max_prob_map and type
            update_mask = mask & ((np.isnan(max_prob_map)) | (prob_map > max_prob_map))
            max_prob_map[update_mask] = prob_map[update_mask]
            max_prob_map_type[update_mask] = anomaly_type

        # Step 2: Thresholding, find out where max probability is above threshold, for those are not nan less than threshold,
        # give them a prior value
        filtered_prob_map = np.where( (max_prob_map >= self.pro_threshold) & ~np.isnan(max_prob_map),
                                      max_prob_map, self.prior)


        # Step 3: Connected component labeling per anomaly type
        region_id_map = np.zeros_like(max_prob_map, dtype=np.int32)
        next_region_id = 1 # TODO verify and understand this logic, it is used to assign unique IDs to connected regions
        for anomaly_type in self.road_anomaly_types:
            mask = (max_prob_map_type == anomaly_type)
            # Label connected regions (4-connectivity)
            labeled, num_features = label(mask)
            # Offset region IDs to make each unique
            labeled[labeled > 0] += next_region_id - 1
            region_id_map += labeled
            next_region_id += num_features

        avg_prob_map = None
        if average_neighbors:
            avg_prob_map = np.full_like(max_prob_map, np.nan)
            for region_id in np.unique(region_id_map):
                if region_id == 0:
                    continue
                region_mask = (region_id_map == region_id)
                mean_val = np.nanmean(filtered_prob_map[region_mask])
                avg_prob_map[region_mask] = mean_val

        return filtered_prob_map, max_prob_map_type, region_id_map, avg_prob_map


    @staticmethod
    def _prob_to_logodds(p):
        return np.log(p / (1.0 - p))

    @staticmethod
    def _logodds_to_prob(l):
        return 1.0 - 1.0 / (1.0 + np.exp(l))



if __name__ == "__main__":

    NET_FILE = "scenario/brussel_rural/osm.net.xml"
    # NET_FILE = "scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml"
    ROAD_ANOMALY_METRICS_FILE = "data/road_anomaly_metrics.json"
    PROB_DICT = load_road_anomaly_metrics(ROAD_ANOMALY_METRICS_FILE)
    GPS_SIGMA = 5.0  # GPS noise in meters
    sensor = VehicleSensor(PROB_DICT, GPS_SIGMA, None),
    # Create grid (bounds auto-computed)
    grid = OccupancyGrid(
        net_file=NET_FILE,
        sensor=PROB_DICT,
        resolution=1.0,
        prior=0.1,
    )

    # visualize lane centerlines
    grid.visualize_lane_centerline(NET_FILE)

    # Get occupancy probability map
    prob_map_dict = grid.gen_probability_map()

    # Plot
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # NaN = transparent/white

    # plot the probability map for each road anomaly type
    for anomaly_type, prob_map in prob_map_dict.items():
        masked = np.ma.masked_invalid(prob_map)  # mask NaNs

        img = plt.imshow(
            masked,
            cmap=cmap,
            origin='lower',
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
            interpolation='nearest'
        )

        plt.colorbar(img, label='Occupancy Probability')
        plt.title("Occupancy Grid Map for " + anomaly_type)
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(True, linestyle=':', linewidth=0.5)

        # Save image
        plt.savefig("image/" + anomaly_type + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    # Plot the maximum probability map
    filtered_prob_map, max_prob_map_type, region_id_map, avg_prob_map = grid.filter_results(average_neighbors=True)

    plt.figure(figsize=(12, 10))
    masked = np.ma.masked_where(np.isnan(filtered_prob_map), filtered_prob_map)
    img = plt.imshow(
        masked,
        cmap='viridis',
        origin='lower',
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    )
    plt.colorbar(img, label='Filtered Max Occupancy Probability')
    plt.title("Filtered Max Occupancy Grid Map")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.savefig("image/filtered_max_occupancy_grid_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    import matplotlib.colors as mcolors

    # Assign each anomaly type a unique integer
    type_names = list(set(max_prob_map_type[max_prob_map_type != '']))
    type_to_int = {typ: i for i, typ in enumerate(type_names)}
    type_map = np.full_like(filtered_prob_map, -1, dtype=int)
    for typ, i in type_to_int.items():
        type_map[max_prob_map_type == typ] = i

    cmap = plt.get_cmap('tab20', len(type_names))

    plt.figure(figsize=(12, 10))
    masked_type_map = np.ma.masked_where(type_map == -1, type_map)
    img = plt.imshow(
        masked_type_map,
        cmap=cmap,
        origin='lower',
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        vmin=0, vmax=len(type_names) - 1
    )
    # Create legend
    cbar = plt.colorbar(img, ticks=np.arange(len(type_names)))
    cbar.ax.set_yticklabels(type_names)
    plt.title("Maximum Probability Anomaly Type Map")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.savefig("image/max_prob_type_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    plt.figure(figsize=(12, 10))
    region_ids = region_id_map.copy()
    # Don't hide background, showisng the road region
    region_ids[region_ids == 0] = np.nan
    n_regions = int(np.nanmax(region_ids)) + 1
    cmap_regions = cm.get_cmap('tab20', n_regions)

    img = plt.imshow(
        region_ids,
        cmap=cmap_regions,
        origin='lower',
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
    )
    plt.colorbar(img, label='Region ID')
    plt.title("Connected Region ID Map (per Anomaly Type)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.savefig("image/region_id_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    if avg_prob_map is not None:
        plt.figure(figsize=(12, 10))
        masked_avg = np.ma.masked_where(np.isnan(avg_prob_map), avg_prob_map)
        img = plt.imshow(
            masked_avg,
            cmap='plasma',
            origin='lower',
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max]
        )
        plt.colorbar(img, label='Region-averaged Probability')
        plt.title("Region-Averaged Probability Map")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.savefig("image/region_averaged_probability_map.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()






