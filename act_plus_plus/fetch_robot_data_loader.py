import cv2
import unittest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image


class FetchRobotDataset(Dataset):
    cache = {}
    file_cache = {}

    def __init__(self, demo_dir, traj_len=1, is_train=True, use_ori_pil_img=False, use_data_aug=False):
        super(FetchRobotDataset, self).__init__()
        self.use_ori_pil_img = use_ori_pil_img
        self.use_data_aug = use_data_aug
        self.demo_dir = demo_dir
        self.traj_len = traj_len
        self.demo_files = glob.glob(f"{demo_dir}/episode_*")
        self.demo_files.sort()
        # self.get_norm_stats()
        line = int(len(self.demo_files) * 0.9)
        if is_train:
            self.demo_files = self.demo_files[:line]
        else:
            self.demo_files = self.demo_files[line:]
        self.is_sim = True

        # create index map
        self.index_map = None
        for file_idx, _ in enumerate(self.demo_files):
            joint_states = np.load(f"{self.demo_dir}/episode_{file_idx}/joint_states.npy").astype(np.float32)
            step_index_list = np.array(range(joint_states.shape[0]))
            step_index_list = step_index_list[:, None].repeat(2, axis=1)
            step_index_list[:, 0] = file_idx
            if self.index_map is None:
                self.index_map = step_index_list
            else:
                self.index_map = np.concatenate((self.index_map, step_index_list), axis=0)

    def get_norm_stats(self):
        return None

    def __len__(self):
        return self.index_map.shape[0]

    def __getitem__(self, idx):
        file_idx = self.index_map[idx, 0]
        step_idx = self.index_map[idx, 1]

        if file_idx in FetchRobotDataset.file_cache:
            joints = FetchRobotDataset.file_cache[file_idx]
            controller_btn_states = FetchRobotDataset.file_cache[f"{file_idx}_btn"]
            acml_poses = FetchRobotDataset.file_cache[f"{file_idx}_acml"]
            stop_signals = FetchRobotDataset.file_cache[f"{file_idx}_stop"]
        else:
            joints = np.load(f"{self.demo_dir}/episode_{file_idx}/joint_states.npy").astype(np.float32)
            FetchRobotDataset.file_cache[file_idx] = joints
            controller_btn_states = np.load(f"{self.demo_dir}/episode_{file_idx}/controller_btn_states.npy").astype(
                np.float32)
            FetchRobotDataset.file_cache[f"{file_idx}_btn"] = controller_btn_states
            acml_poses = np.load(f"{self.demo_dir}/episode_{file_idx}/acml_poses.npy").astype(np.float32)
            FetchRobotDataset.file_cache[f"{file_idx}_acml"] = acml_poses
            stop_signals = np.load(f"{self.demo_dir}/episode_{file_idx}/stop_signal.npy").astype(np.float32)
            FetchRobotDataset.file_cache[f"{file_idx}_stop"] = stop_signals

        controller_btn_states = np.concatenate((controller_btn_states,
                                                controller_btn_states[-1][None, ...]), axis=0)

        # Convert lists to torch tensors
        target_joints = torch.tensor(joints[step_idx:step_idx + self.traj_len], dtype=torch.float32)
        target_btn_states = torch.tensor(controller_btn_states[step_idx:step_idx + self.traj_len], dtype=torch.float32)
        target_acml_poses = torch.tensor(acml_poses[step_idx:step_idx + self.traj_len], dtype=torch.float32)
        target_stop_signals = torch.tensor(stop_signals[step_idx:step_idx + self.traj_len], dtype=torch.float32)
        if target_joints.shape[0] < self.traj_len:
            target_joints = torch.cat(
                (target_joints, torch.zeros(self.traj_len - target_joints.shape[0], target_joints.shape[1])))
            target_btn_states = torch.cat(
                (target_btn_states, torch.zeros(self.traj_len - target_btn_states.shape[0], target_btn_states.shape[1])))
            target_acml_poses = torch.cat(
                (target_acml_poses, torch.zeros(self.traj_len - target_acml_poses.shape[0], target_acml_poses.shape[1])))
            target_stop_signals = torch.cat(
                (target_stop_signals, torch.zeros(self.traj_len - target_stop_signals.shape[0])))

        # concate
        target_stop_signals = target_stop_signals.unsqueeze(1)
        stop_signals = stop_signals[:, None]
        target_joints = torch.cat((target_joints, target_btn_states, target_acml_poses, target_stop_signals), dim=1)
        joints = np.concatenate((joints, controller_btn_states, acml_poses, stop_signals), axis=1)
        joints = torch.tensor(joints[step_idx], dtype=torch.float32)

        # generate target padding mask for transformer decoder
        target_padding_mask = torch.zeros(self.traj_len, dtype=torch.bool)
        target_padding_mask[target_joints[:, 0] == 0] = True

        if self.use_ori_pil_img:
            img_path = f"{self.demo_dir}/episode_{file_idx}/rgb_{step_idx}.jpg"
            rgb = Image.open(img_path)
            
        else:
            if f"episode_{file_idx}/rgb_{step_idx}.jpg" in FetchRobotDataset.cache:
                rgb = FetchRobotDataset.cache[f"episode_{file_idx}/rgb_{step_idx}.jpg"]
                depth = FetchRobotDataset.cache[f"episode_{file_idx}/depth_{step_idx}.jpg"]
            else:
                with open(f"{self.demo_dir}/episode_{file_idx}/rgb_{step_idx}.jpg", "rb") as f:
                    rgb = f.read()
                with open(f"{self.demo_dir}/episode_{file_idx}/depth_{step_idx}.jpg", "rb") as f:
                    depth = f.read()
                FetchRobotDataset.cache[f"episode_{file_idx}/rgb_{step_idx}.jpg"] = rgb
                FetchRobotDataset.cache[f"episode_{file_idx}/depth_{step_idx}.jpg"] = depth
            rgb = np.frombuffer(rgb, np.uint8)
            rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
            if self.use_data_aug:
                rgb = self.data_augmentation(rgb)
            depth = np.frombuffer(depth, np.uint8)
            depth = cv2.imdecode(depth, cv2.IMREAD_COLOR)

            imgs = [rgb, depth]
            imgs = np.array(imgs)
            rgb = torch.from_numpy(imgs) / 255.0
            rgb = rgb.permute(0, 3, 1, 2)  # (480, 640, 3) -> (3, 480, 640)

            # target_joints = (target_joints - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            # joints = (joints - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return rgb, joints, target_joints, target_padding_mask


class FetchRobotDataLoader:
    def __init__(self, demo_dir, k, batch_size=64, num_workers=4):
        self.dataset = FetchRobotDataset(demo_dir, k)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                     pin_memory=True, persistent_workers=True)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    demo_dir = "env/expert_demo/open_microwave"  # The directory where your demo data is stored.
    k = 360
    dataloader = FetchRobotDataLoader(demo_dir, k=k)

    # Test if first batch contains the correct keys
    first_batch = next(iter(dataloader))

    print("done")
