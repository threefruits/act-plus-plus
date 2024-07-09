import cv2
import jax

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

import unittest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import glob


class Fold3Dataset(Dataset):
    def __init__(self, demo_dir, k=None, is_train=True):
        super(Fold3Dataset, self).__init__()
        self.demo_dir = demo_dir
        self.demo_files = glob.glob(os.path.join(f"{demo_dir}/raw", "*.pkl"))
        self.demo_files.sort()
        self.k = k
        self.traj_len = 120
        # self.pre_process()
        self.demo_files = glob.glob(f"{demo_dir}/demo_*")
        self.demo_files.sort()
        self.get_norm_stats()
        line = int(len(self.demo_files) * 0.9)
        if is_train:
            self.demo_files = self.demo_files[:line]
        else:
            self.demo_files = self.demo_files[line:]
        self.cache = {}
        self.file_cache = {}
        self.is_sim = True

    def pre_process(self):
        from core.env.fold3 import Fold3Env
        env = Fold3Env(batch_size=1)

        for file_path in self.demo_files:
            file_name = os.path.basename(file_path)
            target_file_path = f"{self.demo_dir}/{file_name[:-4]}"
            if os.path.exists(target_file_path):
                continue
            os.makedirs(target_file_path, exist_ok=True)
            print(f"Pre-processing {file_path}")

            demo_np = {}
            with open(file_path, "rb") as file:
                demo = pickle.load(file)
                # demo_np["action"] = np.array(demo["action"])

                joints = []
                for i in range(len(demo["state"])):
                    state = demo["state"][i]
                    joint = np.array(state.primitive0).flatten()
                    suction = np.array(state.action0)[0, 3]
                    joint[3] = suction
                    joints.append(joint)

                    # rgb
                    rgbs, _ = env.get_obs_joint(state)
                    # save rgb images to disk

                    for j, rgb in enumerate(rgbs):
                        save_path = f"{self.demo_dir}/{file_name[:-4]}/{file_name[:-4]}_{i}_{j}.jpg"
                        # compress image without reduce dimension and save to disk
                        # rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(save_path, rgb)

                demo_np["joints"] = np.array(joints)
                np.save(f"{self.demo_dir}/{file_name[:-4]}/joints.npy", demo_np["joints"])

            # save the pre-processed demo
            # with open(target_file_path, "wb") as file:
            #     pickle.dump(demo_np, file)

    def get_norm_stats(self):
        all_qpos_data = []
        all_action_data = []
        for episode_path in self.demo_files:
            action = np.load(f"{episode_path}/joints.npy")
            qpos = action
            all_qpos_data.append(torch.from_numpy(action))
            all_action_data.append(torch.from_numpy(action))

        all_qpos_data = torch.stack(all_qpos_data)
        all_action_data = torch.stack(all_action_data)
        all_action_data = all_action_data

        # normalize action data
        action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
        action_std = all_action_data.std(dim=[0, 1], keepdim=True)
        action_std = torch.clip(action_std, 1e-3, 10)  # clipping

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-3, 10)  # clipping

        stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
                 "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
                 "example_qpos": qpos}
        self.norm_stats = stats
        return stats

    def __len__(self):
        return len(self.demo_files) * self.traj_len

    def __getitem__(self, idx):
        file_idx = idx // self.traj_len
        step_idx = idx % self.traj_len

        joints = np.load(f"{self.demo_dir}/demo_{file_idx}/joints.npy")
        self.k = joints.shape[0]

        # Convert lists to torch tensors
        target_joints = torch.tensor(joints[step_idx:step_idx + self.k], dtype=torch.float32)
        if target_joints.shape[0] < self.k:
            target_joints = torch.cat((target_joints, torch.zeros(self.k - target_joints.shape[0], 4)))

        # generate target padding mask for transformer decoder
        target_padding_mask = torch.zeros(self.k, dtype=torch.bool)
        target_padding_mask[target_joints[:, 0] == 0] = True
        joints = torch.tensor(joints[step_idx], dtype=torch.float32)

        img_files = glob.glob(f"{self.demo_dir}/demo_{file_idx}/demo_{file_idx}_{step_idx}_*.jpg")
        imgs = [cv2.imread(img_file) for img_file in img_files]
        imgs = np.array(imgs)
        rgb = torch.from_numpy(imgs) / 255.0
        rgb = rgb.permute(0, 3, 1, 2)  # (480, 640, 3) -> (3, 480, 640)

        # target_joints = (target_joints - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        # joints = (joints - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return rgb, joints, target_joints, target_padding_mask


class Fold3DataLoader:
    def __init__(self, demo_dir, k, batch_size=64, num_workers=4):
        self.dataset = Fold3Dataset(demo_dir, k)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                     pin_memory=True, persistent_workers=True)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class TestDataLoader(unittest.TestCase):
    def test_dataloader(self):
        demo_dir = "../envs/expert_demo/fold_cloth3"  # The directory where your demo data is stored.
        k = 50
        dataloader = Fold3DataLoader(demo_dir, k=k)

        # Test if DataLoader is not empty
        self.assertTrue(len(dataloader) > 0)

        # Test if first batch contains the correct keys
        first_batch = next(iter(dataloader))
        self.assertIn('action', first_batch)
        self.assertIn('joints', first_batch)
        self.assertIn('rgb', first_batch)

        # Test if first batch contains data of correct types
        self.assertTrue(isinstance(first_batch['action'], torch.Tensor))
        self.assertTrue(isinstance(first_batch['joints'], torch.Tensor))
        self.assertTrue(isinstance(first_batch['rgb'], torch.Tensor))

        # Test if first batch contains data of correct shapes (assuming action: (4,), state: (4,), rgb: (480, 640, 3))
        self.assertEqual(first_batch['action'].shape[1:], torch.Size([k, 4]))
        self.assertEqual(first_batch['joints'].shape[1:], torch.Size([4]))
        self.assertEqual(first_batch['rgb'].shape[1:], torch.Size([1, 3, 480, 640]))
