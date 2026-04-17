import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import proj3d

plt.style.use('ggplot')

class Arrow3D(FancyArrowPatch):
    """
    A custom 3D arrow using FancyArrowPatch.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        proj = self.axes.get_proj()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, proj)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, proj=None):
        if proj is None:
            proj = self.axes.get_proj()
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, proj)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def load_camera_data(json_file):
    """
    Loads camera data from a Nerfstudio-style transforms.json file.
    Expects a "frames" key with a list of entries, each containing:
      - "transform_matrix": a 4x4 matrix (list of lists)
    
    Returns a list of dictionaries with keys:
      "position": 3D translation (numpy array),
      "transform": the full 4x4 transform (numpy array),
      "direction": camera viewing direction (numpy array)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    cameras = []
    for frame in data.get('frames', []):
        transform = np.array(frame['transform_matrix'])
        position = transform[:3, 3]
        R = transform[:3, :3]
        direction = -R[:, 2]

        # Extract camera number from filename (e.g., frame_00022.jpg -> 22)
        file_path = frame["file_path"]
        camera_number = int(file_path.split("_")[-1].split(".")[0])

        cameras.append({
            "position": position,
            "transform": transform,
            "direction": direction,
            "camera_number": camera_number
        })

    return cameras

def create_camera_marker():
    """
    Create a custom marker in the shape of a simple camera icon using a matplotlib Path.
    """
    vertices = [
        (0.0, 0.2),  # bottom left of camera body
        (0.0, 0.8),  # top left
        (0.2, 0.8),  # start of notch
        (0.3, 1.0),  # notch tip (flash area)
        (0.7, 1.0),  # notch tip right
        (0.8, 0.8),  # end of notch
        (1.0, 0.8),  # top right
        (1.0, 0.2),  # bottom right
        (0.0, 0.2),  # close path
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    return Path(vertices, codes)

def plot_and_save_cameras(cameras, arrow_length=0.4,
                          default_save_path="camera_positions_professional.png",
                          top_view_save_path="camera_positions_topview.png"):
    """
    Plots camera positions with custom camera-like markers and stylish red 3D arrows for viewing directions.
    Saves two images: one from the default view and one from a top view.
    
    Parameters:
      cameras: list of dicts with camera data.
      arrow_length: length multiplier for the viewing direction arrow.
      default_save_path: filename to save the default view.
      top_view_save_path: filename to save the top view.
    """
    # Create the figure and axis.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    camera_marker = create_camera_marker()
    
    # Gather all camera positions for setting axis limits.
    all_positions = np.array([cam["position"] for cam in cameras])
    
    for cam in cameras:
        pos = cam["position"]
        direction = cam["direction"]
        ax.scatter(pos[0], pos[1], pos[2],
                   marker=camera_marker, s=200, c='black', edgecolors='white', zorder=5)
        norm_dir = direction / np.linalg.norm(direction)
        end = pos + arrow_length * norm_dir
        arrow = Arrow3D([pos[0], end[0]],
                        [pos[1], end[1]],
                        [pos[2], end[2]],
                        mutation_scale=20, lw=2, arrowstyle="-|>", color="red")
        ax.add_artist(arrow)
        
        cam_num = cam.get("camera_number", -1)
        offset = 0.5  # you can tweak this for clarity
        label_pos = pos - offset * norm_dir
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                f"{cam_num}", fontsize=8, color='blue', ha='center')


    
    ax.set_title(f"{FRUIT} (GT): Camera Positions and Viewing Directions", fontsize=14, fontweight='bold')
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.zaxis.set_label_coords(-0.05, 0.5)   # (x, y) tweak until inside the gray area

    
    # Set equal aspect ratio.
    max_range = (all_positions.max(axis=0) - all_positions.min(axis=0)).max() / 2.0
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save default view.
    plt.savefig(default_save_path, dpi=300, bbox_inches='tight', pad_inches=0.4)
    print(f"Default view saved to {default_save_path}")
    
    # Change view to top view (looking down, elev=90)
    ax.view_init(elev=90, azim=-90)
    # ax.set_title(f"Honeycrisp_SC - Camera Positions (top-view)", fontsize=14, fontweight='bold')
    plt.savefig(top_view_save_path, dpi=300, bbox_inches='tight', pad_inches=0.4)
    print(f"Top view saved to {top_view_save_path}")
    
    plt.show()


if __name__ == '__main__':
    
    global FRUIT
    
    #----- Replace with your actual path
    FRUIT = "{FRUIT_NAME}"  
    PAR_DIR = "{PARENT_DIR_PATH}"
    FRUIT_DIR = "{PAR_DIR}/{FRUIT}"
    JSON_FILE = "{FRUIT_DIR}/transforms.json"
    
    cameras = load_camera_data(JSON_FILE)
    
    print(f"Loaded {len(cameras)} cameras.")
    
    plot_and_save_cameras(cameras, arrow_length=1,
                        default_save_path= f"{FRUIT_DIR}/isometric-view.png",
                        top_view_save_path= f"{FRUIT_DIR}/top-view.png")
    

