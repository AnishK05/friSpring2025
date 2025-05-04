import h5py
import numpy as np
import torch

def load_hdf5(path):
    demonstrations = []
    pose_stats = {"min" : float('inf'), "max" : float('-inf')}
    orientation_stats = {"min" : float('inf'), "max" : float('-inf')}
    iq_stats = {"min" : float('inf'), "max" : float('-inf')}
    color_stats = {"min" : float('inf'), "max" : float('-inf')}
    depth_stats = {"min" : float('inf'), "max" : float('-inf')}

    with h5py.File(path, "r") as f:
        data_group = f["data"]
        for demo_key in data_group:
            demo_group = data_group[demo_key]
            num_samples = demo_group.attrs["num_samples"]

            colors = demo_group["obs/color"][:num_samples]         # (T, H, W, 3)
            depths = demo_group["obs/depth"][:num_samples]         # (T, H, W)
            states = demo_group["obs/states"][:num_samples]        # compound dtype

            demo = []
            for i in range(num_samples):
                state = states[i]  # shape (8,)
                demo.append({
                    "color": np.transpose(colors[i], (2, 0, 1)),        # (3, H, W)
                    "depth": np.transpose(depths[i], (2, 0, 1)),               # (1, H, W)
                    "position": state[:3][np.newaxis, :],              # (1, 3)
                    "orientation": state[3:7][np.newaxis, :],          # (1, 4)
                    "iq": state[7:][np.newaxis, :]                     # (1, 1)
                })

            demonstrations.append(demo)
    if len(demonstrations) == 0:
        print("No demonstrations found")
    elif len(demonstrations) == 1:
        print(f"Loaded 1 demonstration from {path}")
    else:
        print(f"Loaded {len(demonstrations)} demonstrations from {path}")
    return demonstrations

def get_stats(demonstrations):
    pos_min, pos_max = None, None
    or_min, or_max = None, None
    iq_min, iq_max = None, None
    color_min, color_max = float("inf"), float("-inf")
    depth_min, depth_max = float("inf"), float("-inf")

    for demo in demonstrations:
        for sample in demo:
            pos = torch.from_numpy(sample["position"]).squeeze(0)
            ori = torch.from_numpy(sample["orientation"]).squeeze(0)
            iq = torch.from_numpy(sample["iq"]).squeeze(0)

            pos_min = pos if pos_min is None else torch.minimum(pos_min, pos)
            pos_max = pos if pos_max is None else torch.maximum(pos_max, pos)
            or_min = ori if or_min is None else torch.minimum(or_min, ori)
            or_max = ori if or_max is None else torch.maximum(or_max, ori)
            iq_min = iq if iq_min is None else torch.minimum(iq_min, iq)
            iq_max = iq if iq_max is None else torch.maximum(iq_max, iq)

            color = torch.from_numpy(sample["color"]).float()
            depth = torch.from_numpy(sample["depth"]).float()

            color_min = min(color_min, color.min().item())
            color_max = max(color_max, color.max().item())
            depth_min = min(depth_min, depth.min().item())
            depth_max = max(depth_max, depth.max().item())
    print("Successfully loaded stats")
    return { # pose
            "min": pos_min,
            "max": pos_max
        }, { # orientation
            "min": or_min,
            "max": or_max
        }, { # iq
            "min": iq_min,
            "max": iq_max
        }, { # color
            "min": color_min,
            "max": color_max
        }, { # depth
            "min": depth_min,
            "max": depth_max
        }