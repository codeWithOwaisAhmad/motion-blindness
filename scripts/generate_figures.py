import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure clean font rendering
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'dejavuserif'
})

VERIFY_DIR = "verify_frames"
if not os.path.isdir(VERIFY_DIR):
    VERIFY_DIR = "/home/owais/Documents/motion-blindness/data/verify"

# Function to get 4 uniform frames from a clip
def get_4_frames(clip_id):
    cdir = os.path.join(VERIFY_DIR, clip_id)
    rgbs = sorted(glob.glob(os.path.join(cdir, "*_rgb.jpg")))
    depths = sorted(glob.glob(os.path.join(cdir, "*_depth.jpg")))
    if not rgbs:
        files = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
        rgbs = [f for f in files if "depth" not in f.lower()]
        depths = [f for f in files if "depth" in f.lower()]
    
    # Filter initial warmup frame
    rgbs = rgbs[1:] if len(rgbs) >= 5 else rgbs
    depths = depths[1:] if len(depths) >= 5 else depths
    
    idx = np.linspace(0, len(rgbs) - 1, 4, dtype=int)
    rgb_imgs = [cv2.cvtColor(cv2.imread(rgbs[i]), cv2.COLOR_BGR2RGB) for i in idx]
    depth_imgs = [cv2.cvtColor(cv2.imread(depths[i]), cv2.COLOR_BGR2RGB) for i in idx]
    return rgb_imgs, depth_imgs

# Select two clean, pure object clips (Approach vs. Recession)
# clip_002: Radial approach (Z: 1.22m -> 0.39m)
# clip_064 or clip_056: Diagonal/Radial recession (Z: 0.39m -> 1.22m)
rgb_app, depth_app = get_4_frames("clip_002")
rgb_rec, depth_rec = get_4_frames("clip_045")

fig = plt.figure(figsize=(13, 8.5), dpi=300)
gs = GridSpec(5, 5, height_ratios=[1, 1, 0.15, 1, 1], width_ratios=[1, 1, 1, 1, 0.04], hspace=0.18, wspace=0.06)

time_stamps = ["t = 0.0 s", "t = 0.8 s", "t = 1.6 s", "t = 2.4 s"]

# ----------------- SECTION 1: APPROACH SEQUENCE -----------------
for col in range(4):
    # RGB Row
    ax_rgb = fig.add_subplot(gs[0, col])
    ax_rgb.imshow(rgb_app[col])
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
    ax_rgb.set_title(time_stamps[col], fontsize=9, pad=5)
    for spine in ax_rgb.spines.values():
        spine.set_color('#444444'); spine.set_linewidth(0.8)
    if col == 0:
        ax_rgb.set_ylabel("Trajectory I: Approach\nRGB Stream", fontsize=9, fontweight='bold')

    # Depth Row
    ax_dep = fig.add_subplot(gs[1, col])
    ax_dep.imshow(depth_app[col])
    ax_dep.set_xticks([]); ax_dep.set_yticks([])
    for spine in ax_dep.spines.values():
        spine.set_color('#444444'); spine.set_linewidth(0.8)
    if col == 0:
        ax_dep.set_ylabel("RealSense D435i\nActive IR Depth", fontsize=9, fontweight='bold')

# ----------------- SECTION 2: RECESSION SEQUENCE -----------------
for col in range(4):
    # RGB Row
    ax_rgb = fig.add_subplot(gs[3, col])
    ax_rgb.imshow(rgb_rec[col])
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
    ax_rgb.set_title(time_stamps[col], fontsize=9, pad=5)
    for spine in ax_rgb.spines.values():
        spine.set_color('#444444'); spine.set_linewidth(0.8)
    if col == 0:
        ax_rgb.set_ylabel("Trajectory II: Recession\nRGB Stream", fontsize=9, fontweight='bold')

    # Depth Row
    ax_dep = fig.add_subplot(gs[4, col])
    ax_dep.imshow(depth_rec[col])
    ax_dep.set_xticks([]); ax_dep.set_yticks([])
    for spine in ax_dep.spines.values():
        spine.set_color('#444444'); spine.set_linewidth(0.8)
    if col == 0:
        ax_dep.set_ylabel("RealSense D435i\nActive IR Depth", fontsize=9, fontweight='bold')

# ----------------- DEPTH COLORBAR (COLUMN 5) -----------------
cbar_ax = fig.add_subplot(gs[:, 4])
gradient = np.linspace(0.3, 3.0, 256).reshape(256, 1)
cbar_ax.imshow(gradient, aspect='auto', cmap='jet_r', extent=[0, 1, 0.3, 3.0])
cbar_ax.set_xticks([])
cbar_ax.set_ylabel("Metric Depth $Z$ (Meters)", fontsize=10, labelpad=10)
cbar_ax.yaxis.set_label_position("right")
cbar_ax.yaxis.tick_right()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig1_teaser_filmstrip.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved clean, professional Figure 1 to figures/fig1_teaser_filmstrip.png")