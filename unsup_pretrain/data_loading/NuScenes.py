import importlib
import os
import sys

import numpy as np

from data_loading.ipb_base import IPB_Base
from torchvision import transforms
import torch


class NuScenes(IPB_Base):

    def __init__(self,
                 data_source=None,
                 path_in_dataserver='~/data/nuScenes/v1.0-mini.tgz',
                 sequence: int = 61,
                 max_range=100.0,
                 min_range=5.0,
                 split='mini_train',
                 unittesting=False,
                 overfit=False):
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed on your system")
            print('run "pip install nuscenes-devkit"')
            sys.exit(1)

        super().__init__(data_source, path_in_dataserver, overfit, unittesting)

        # TODO: If someone needs more splits from nuScenes expose this 2 parameters
        #  nusc_version: str = "v1.0-trainval"
        #  split: str = "train"
        nusc_version: str = "v1.0-mini"
        split: str = split
        self.lidar_name: str = "LIDAR_TOP"
        self.camera_name: str = "CAM_FRONT"

        # Lazy loading
        from nuscenes.nuscenes import NuScenes as NS
        from nuscenes.utils.splits import create_splits_logs

        # Config stuff
        self.sequence_id = str(int(sequence)).zfill(4)
        self.min_range = min_range
        self.max_range = max_range

        self.nusc = NS(dataroot=str(data_source), version=nusc_version)
        self.scene_name = f"scene-{self.sequence_id}"
        if self.scene_name not in [s["name"] for s in self.nusc.scene]:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            self.nusc.list_scenes()
            sys.exit(1)

        # Load nuScenes read from file inside dataloader module
        self.load_point_cloud = importlib.import_module("nuscenes.utils.data_classes").LidarPointCloud.from_file

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, self.nusc)

        # Use only the samples from the current split.
        scene_token = self._get_scene_token(split_logs)
        self.lidar_tokens = self._get_lidar_tokens(scene_token)
        self.camera_tokens = self._get_camera_tokens(scene_token)
        self.gt_poses = self._load_poses()

    def __len__(self):
        return len(self.camera_tokens)

    def __getitem__(self, idx):
        lidar_token = self.lidar_tokens[idx]
        filename = self.nusc.get("sample_data", lidar_token)["filename"]
        pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, filename))
        points = pcl.points.T[:, :3]
        pose = self.gt_poses[idx]
        camera_token = self.camera_tokens[idx]
        cropped_points, coloring, im = self.nusc.explorer.map_pointcloud_to_image(lidar_token, camera_token,
                                                                                  render_intensity=True)
        depth_im = np.zeros([90, 160])
        reduced_cropped_points = (cropped_points / 10).astype(int)
        depth_im[reduced_cropped_points[1].astype(int), reduced_cropped_points[0].astype(int)] = coloring

        # import matplotlib.pyplot as plt
        # plt.imshow(depth_im)
        # plt.show()

        transform_image = transforms.Compose(
            [
                transforms.Resize((90, 160), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        im = transform_image(im).type(torch.float32)
        depth_im = transforms.ToTensor()(depth_im).type(torch.float32)

        cat_im = torch.cat((im, depth_im), 0)

        # item = {'points': points.astype(np.float64), 'pose': pose, 'filename': filename, 'cropped_points': depth_im,
        #         'image': im}

        # return item

        return  cat_im
    def _preprocess(self, points):
        points = points[np.linalg.norm(points, axis=1) <= self.max_range]
        points = points[np.linalg.norm(points, axis=1) >= self.min_range]
        return points

    def _load_poses(self) -> np.ndarray:
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        poses = np.empty((len(self), 4, 4), dtype=np.float32)
        for i, lidar_token in enumerate(self.lidar_tokens):
            sd_record_lid = self.nusc.get("sample_data", lidar_token)
            cs_record_lid = self.nusc.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
            ep_record_lid = self.nusc.get("ego_pose", sd_record_lid["ego_pose_token"])

            car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            )
            pose_car = transform_matrix(
                ep_record_lid["translation"],
                Quaternion(ep_record_lid["rotation"]),
            )

            poses[i:, :] = pose_car @ car_to_velo

        # Convert from global coordinate poses to local poses
        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses
        return poses

    def _get_scene_token(self, split_logs):
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = [s["token"] for s in self.nusc.scene if s["name"] == self.scene_name][0]
        scene = self.nusc.get("scene", scene_tokens)
        log = self.nusc.get("log", scene["log_token"])
        return scene["token"] if log["logfile"] in split_logs else ""

    def _get_lidar_tokens(self, scene_token: str):

        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", start_sample_rec["data"][self.lidar_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sdl_tokens = []
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get("sample_data", cur_sd_rec["next"])
            sdl_tokens.append(cur_sd_rec["token"])
        return sdl_tokens

    def _get_camera_tokens(self, scene_token: str):

        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", start_sample_rec["data"][self.camera_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sdc_tokens = []
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get("sample_data", cur_sd_rec["next"])
            sdc_tokens.append(cur_sd_rec["token"])
        return sdc_tokens

if __name__ == '__main__':
    kitti_dataset = NuScenes(root_dir='/home/matt/data/nuScenes/')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for i, sample in enumerate(kitti_dataset):
        print(i, sample)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        if i == 3:
            plt.show()
            break
