import os
import glob

import fiftyone as fo
import fiftyone.utils.random as fo_rng
import numpy as np
import open3d as o3d    #for lidar data

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from huggingface_hub import HfApi

api = HfApi()

# --- CONFIG ---
LIDAR_TYPE = ".pcd"
IMAGE_TYPE = ".png"

classes = ["cubes", "spheres"]
data_root = "../data/assessment"
# --------------


def _spherical_to_xyz(
    lidar_depth: np.ndarray,
    azimuth: np.ndarray,
    zenith: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a (64,64) LiDAR depth map to Cartesian (x, y, z) numpy arrays.

    Single source of truth for the spherical coordinate transform used by both
    npy_to_pcd (visualization) and get_torch_xyza (training tensors).
    Negative angles align LiDAR orientation with the RGB image frame.
    azimuth -> rows, zenith -> columns.
    """
    x = lidar_depth * np.sin(-azimuth[:, None]) * np.cos(-zenith[None, :])
    y = lidar_depth * np.cos(-azimuth[:, None]) * np.cos(-zenith[None, :])
    z = lidar_depth * np.sin(-zenith[None, :])
    return x, y, z


def npy_to_pcd():
    """Converts lidar point clouds from .npy to .pcd using azimuth/zenith angle transformation and saves the pcd files in a new directory."""
    for label in classes:
        npy_folder = f"../data/assessment/{label}/lidar/"
        pcd_output_folder = f"../data/assessment/{label}/lidar_{LIDAR_TYPE}/"

        # Load per-class angle arrays (shape: 64,)
        # azimuth -> rows, zenith -> columns
        azimuth = np.load(f"../data/assessment/{label}/azimuth.npy")
        zenith = np.load(f"../data/assessment/{label}/zenith.npy")

        os.makedirs(pcd_output_folder, exist_ok=True)
        npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

        print(f"Found {len(npy_files)} .npy files. Converting...")

        for npy_path in npy_files:
            lidar_depth = np.load(npy_path)  # shape (64, 64)

            x, y, z = _spherical_to_xyz(lidar_depth, azimuth, zenith)

            # Mask out max-range (invalid / no-return) measurements
            valid = lidar_depth != lidar_depth.max()

            points_3d = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            pcd_path = os.path.join(pcd_output_folder, f"{base_name}{LIDAR_TYPE}")
            o3d.io.write_point_cloud(pcd_path, pcd)

    print("Finished npy to pcd conversion.")


# ---- PyTorch helpers ----

IMG_SIZE = 64

_img_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),   # -> (C, H, W) in [0, 1]
])


def get_torch_xyza(lidar_depth: torch.Tensor,
                   azimuth: torch.Tensor,
                   zenith: torch.Tensor) -> torch.Tensor:
    """Convert a (64,64) depth tensor to a (4,64,64) XYZA tensor.

    Delegates to _spherical_to_xyz for the math, then wraps results as tensors.
    Channels: x, y, z (Cartesian positions) + a (validity mask).
    """
    device = lidar_depth.device
    depth_np = lidar_depth.cpu().numpy()
    az_np    = azimuth.cpu().numpy()
    ze_np    = zenith.cpu().numpy()

    x_np, y_np, z_np = _spherical_to_xyz(depth_np, az_np, ze_np)
    a_np = (depth_np < depth_np.max()).astype(np.float32)

    def _t(arr): return torch.from_numpy(arr.astype(np.float32)).to(device)
    return torch.stack([_t(x_np), _t(y_np), _t(z_np), _t(a_np)])  # (4, 64, 64)


class MultimodalDataset(Dataset):
    """Paired RGB + LiDAR dataset built from a FiftyOne grouped dataset.

    The FO dataset is used only for split assignment (train/val tags) and RGB
    file paths.  LiDAR data is loaded from the raw lidar_npy/*.npy files —
    NOT from the lidar_pcd/*.pcd files stored in the FO lidar slice.  The .pcd
    files exist solely for FiftyOne visualization; the .npy files are loaded
    here and converted on-the-fly to XYZA tensors via get_torch_xyza.

    Expects the FO dataset to have:
    - group slices "rgb" and "lidar"
    - split tags "train" / "val" on the rgb slice
    - ground_truth.label in `classes`
    - lidar_npy files at data_root/<class>/lidar_npy/<basename>.npy
    """

    def __init__(self, fo_dataset, split: str,
                 device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.samples = []   # list of (rgb_tensor, xyza_tensor, label_tensor)

        # Cache angle arrays per class (loaded once per class)
        angle_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for class_label in classes:
            az = torch.from_numpy(
                np.load(os.path.join(data_root, class_label, "azimuth.npy"))
            ).float().to(device)
            ze = torch.from_numpy(
                np.load(os.path.join(data_root, class_label, "zenith.npy"))
            ).float().to(device)
            angle_cache[class_label] = (az, ze)

        # Iterate only over RGB slice with the requested split tag
        split_view = fo_dataset.match_tags(split).select_group_slices("rgb")
        for sample in split_view.iter_samples():
            rgb_path  = sample.filepath
            label_str = sample.ground_truth.label   # "cubes" or "spheres"
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            npy_path  = os.path.join(data_root, label_str, "lidar",
                                     base_name + ".npy")

            if not os.path.exists(npy_path):
                print(f"Warning: Corresponding npy fie nt found for {npy_path}")
                continue

            rgb = _img_transforms(
                Image.open(rgb_path).convert("RGB")
            ).to(device)

            lidar_depth = torch.from_numpy(
                np.load(npy_path).astype(np.float32)
            ).to(device)
            az, ze = angle_cache[label_str]
            xyza = get_torch_xyza(lidar_depth, az, ze)

            label = torch.tensor(
                float(classes.index(label_str)), dtype=torch.float32
            ).to(device)
            self.samples.append((rgb, xyza, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_fo_dataset(dataset_name: str, class_fractions: list[float], train_fraction: float = 0.8):
    """Filter total samples for classes (in case of inbalance) and split the dataset. Then Save dataset to fo database."""

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)

    dataset.add_group_field("group", default="rgb")

    samples = []

    rgb_height  = None
    for class_label in classes:
        rgb_paths = glob.glob(os.path.join(data_root, class_label, "rgb", "*"))
        #ASSUMPTION: Each RGB file has a corresponding LiDAR file with the same base name in the "lidar_pcd" folder vice versa.
        for rgb_path in rgb_paths:
            filename =      os.path.basename(rgb_path)
            base_name =     os.path.splitext(filename)[0]
            #fetch corresponding lidar file (could be potentially optimized by doing one sequential read, but for our dataset size it does not matter)
            lidar_path =    os.path.join(data_root, class_label, "lidar_.pcd", base_name + ".pcd")

            group = fo.Group()

            rgb_sample = fo.Sample(filepath=rgb_path, group=group.element("rgb"))
            rgb_sample["ground_truth"] = fo.Classification(label=class_label)
            if rgb_height is None:
                rgb_sample.compute_metadata()
                print("rgb image dimensions are H: ", rgb_sample.metadata.height, " W: ", rgb_sample.metadata.width)
                rgb_height = rgb_sample.metadata.height
            
            lidar_sample = fo.Sample(filepath=lidar_path, group=group.element("lidar"))
            lidar_sample["ground_truth"] = fo.Classification(label=class_label)
            
            samples.extend([rgb_sample, lidar_sample])

    dataset.add_samples(samples)
    #both groups consist of the same number of samples so we can naively split them seperately
    fo_rng.random_split(dataset.select_group_slices("rgb"), {"train": train_fraction, "val": 1-train_fraction}, seed=42)
    fo_rng.random_split(dataset.select_group_slices("lidar"), {"train": train_fraction, "val": 1-train_fraction}, seed=42)
    dataset.save()

    print(f"Dataset created with {len(dataset)} samples and slices {dataset.group_slices} with types {IMAGE_TYPE} and {LIDAR_TYPE}.")
    print(f"Dataset split: {round(train_fraction,2)} train, {round(1-train_fraction,2)} validation.")  


def create_pytorch_dataset_from_fo(
    dataset_name: str,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> tuple[MultimodalDataset, MultimodalDataset, DataLoader, DataLoader]:
    """Load (or create) a FiftyOne dataset and return train/val datasets and dataloaders."""
    if not fo.dataset_exists(dataset_name):
        print(f"ERROR: fo dataset {dataset_name} not found, creating new one...")
        
        # create_fo_dataset(dataset_name, [0.1, 0.1])
    fo_dataset = fo.load_dataset(dataset_name)

    train_ds = MultimodalDataset(fo_dataset, "train", device)
    val_ds   = MultimodalDataset(fo_dataset, "val",   device)
    print(f"Extracted train dataset of size {len(train_ds)} and validation dataset of size {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_ds, val_ds, train_loader, val_loader