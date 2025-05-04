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

    return demonstrations

def get_stats(demonstrations):
    pos_list, or_list, iq_list = [], [], []
    color_min, color_max = float("inf"), float("-inf")
    depth_min, depth_max = float("inf"), float("-inf")

    for demo in demonstrations:
        for sample in demo:
            pos_list.append(torch.from_numpy(sample["position"]).squeeze(0))
            or_list.append(torch.from_numpy(sample["orientation"]).squeeze(0))
            iq_list.append(torch.from_numpy(sample["iq"]).squeeze(0))

            color = torch.from_numpy(sample["color"]).float()
            depth = torch.from_numpy(sample["depth"]).float()

            color_min = min(color_min, color.min().item())
            color_max = max(color_max, color.max().item())
            depth_min = min(depth_min, depth.min().item())
            depth_max = max(depth_max, depth.max().item())

    return { # pose
            "min": torch.stack(pos_list).min(dim=0).values,
            "max": torch.stack(pos_list).max(dim=0).values
        }, { # orientation
            "min": torch.stack(or_list).min(dim=0).values,
            "max": torch.stack(or_list).max(dim=0).values
        }, { # iq
            "min": torch.stack(iq_list).min(dim=0).values,
            "max": torch.stack(iq_list).max(dim=0).values
        }, { # color
            "min": color_min,
            "max": color_max
        }, { # depth
            "min": depth_min,
            "max": depth_max
        }