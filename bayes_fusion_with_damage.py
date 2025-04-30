#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sumolib      # from $SUMO_HOME/tools
import traci

# -----------------------------------------------------------------------------
# 1) Bayesian Occupancy Grid Class (unchanged)
# -----------------------------------------------------------------------------
class BayesianOccupancyGrid:
    def __init__(self, xmin, ymin, xmax, ymax, cell_size=1.0,
                 prior=0.1, p_hit=0.9, p_false=0.05):
        self.cell_size = cell_size
        self.xmin, self.ymin = xmin, ymin
        self.W = int(np.ceil((xmax - xmin) / cell_size))
        self.H = int(np.ceil((ymax - ymin) / cell_size))
        self.P = np.full((self.H, self.W), prior, dtype=np.float32)
        # sensor model (hit/miss)
        self.p_hit      = p_hit
        self.p_miss     = 1 - p_hit
        self.p_false    = p_false
        self.p_truefree= 1 - p_false

    def world_to_cell(self, x, y):
        j = int((x - self.xmin) / self.cell_size)
        i = int((y - self.ymin) / self.cell_size)
        if 0 <= i < self.H and 0 <= j < self.W:
            return i, j
        else:
            return None

    def update_cell(self, i, j, detected):
        prior = self.P[i, j]
        if detected:
            num = self.p_hit * prior
            den = num + self.p_false * (1 - prior)
        else:
            num = self.p_miss * prior
            den = num + self.p_truefree * (1 - prior)
        self.P[i, j] = (num / den) if den > 0 else prior

    def fuse(self, detections):
        for i, j, det in detections:
            self.update_cell(i, j, det)


# -----------------------------------------------------------------------------
# 2) Main simulation + damage integration
# -----------------------------------------------------------------------------
def run_sumo_fusion_with_damage(netfile, routefile, damage_world_pts):
    # start SUMO
    sumo_binary = os.environ.get("SUMO_BINARY", "sumo")
    traci.start([sumo_binary, "-n", netfile, "-r", routefile, "--step-length", "0.001"])

    # read network bounds
    net = sumolib.net.readNet(netfile)
    xmin, ymin, xmax, ymax = net.getBoundary()

    # initialize grid
    grid = BayesianOccupancyGrid(
        xmin, ymin, xmax, ymax,
        cell_size=1.0,   # 1 m
        prior=0.1,       # prior occupancy
        p_hit=0.9,       # TP rate
        p_false=0.05     # FP rate
    )

    # convert damage points to cell indices
    damage_cells = set()
    for (dx, dy) in damage_world_pts:
        cell = grid.world_to_cell(dx, dy)
        if cell:
            damage_cells.add(cell)

    # set up plot (once)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(grid.P, origin="lower", vmin=0, vmax=1, cmap="gray")
    d_i, d_j = zip(*damage_cells) if damage_cells else ([],[])
    scatter = ax.scatter(d_j, d_i, c='red', s=15, label='True Damage')
    ax.legend(loc='upper right')
    ax.set_title("Bayesian Occupancy Grid + True Damage")
    ax.set_xlabel("X cells");  ax.set_ylabel("Y cells")
    plt.show()

    sim_time = 0.0
    last_sec = -1

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time += 0.001
            sec = int(sim_time)

            # only fuse & redraw once per second
            if sec != last_sec:
                last_sec = sec

                # collect & fuse detections (same as before)…
                dets = []
                for vid in traci.vehicle.getIDList():
                    x, y = traci.vehicle.getPosition(vid)
                    cell = grid.world_to_cell(x, y)
                    if cell is None: continue
                    is_damage = (cell in damage_cells)
                    detected  = (np.random.rand() < (grid.p_hit if is_damage else grid.p_false))
                    dets.append((cell[0], cell[1], detected))
                grid.fuse(dets)

                # update occupancy & scatter
                im.set_data(grid.P)
                scatter.set_offsets(np.c_[d_j, d_i])   # note x=j, y=i
                ax.set_title(f"OccGrid + Damage @ t={sec}s")

                # redraw & pause briefly
                fig.canvas.draw_idle()
                plt.pause(0.001)

    finally:
        traci.close()



if __name__ == "__main__":
    NETFILE   = "scenario/A2_GO2GW_v2_MM.net.xml"
    ROUTEFILE = "scenario/A2Graz.rou.xml"
    # --- manually specify your damage locations here: ---
    DAMAGE_WORLD_POINTS = [
        (498.2, 123.5),
        (502.7, 127.1),
        (510.0, 130.0),
        # add as many (x,y) as you like…
    ]
    run_sumo_fusion_with_damage(NETFILE, ROUTEFILE, DAMAGE_WORLD_POINTS)
