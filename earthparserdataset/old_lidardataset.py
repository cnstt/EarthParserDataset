import copy
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_scatter
from numpy.lib.recfunctions import structured_to_unstructured
from torch_geometric.data import Data, InMemoryDataset, Dataset
from tqdm.auto import tqdm

from .base import BaseDataModule
from .transforms import (
    Center,
    CenterCrop,
    GridSampling,
    MaxPoints,
    RealRandomCropTrain,
    RandomCropSubTileTrain,
    RandomCropSubTileVal,
    RandomFlip,
    RandomRotate,
    RandomScale,
    ZCrop
)
from .utils import color as color
from .utils.labels import apply_learning_map, from_sem_to_color


class OldLidarScenes(InMemoryDataset):
    def __init__(self, options, mode):
        self.options = copy.deepcopy(options)
        self.options.data_dir = osp.join(self.options.data_dir, self.options.name)
        self.mode = mode

        self.feature_normalizer = torch.tensor(
            [[self.options.max_xy, self.options.max_xy, self.options.max_z]])

        super().__init__(
            self.options.data_dir,
            transform= self.get_transform(),
            pre_transform= None,
            pre_filter=None
        )
        self.data, self.slices = self.process()

        if mode in ["train", "val"]:
            self.items_per_epoch = int(self.options.items_per_epoch[mode] / (len(self.slices["pos"]) - 1 if self.slices is not None else 1))
        else:
            # TODO: test dataset pipeline
            self.prepare_test_dataset()
    
    def process(self):
        data_list = []
        num_items = self.options.batch_size*self.options.items_per_epoch[self.mode]
        total_processed_paths = len(self.processed_paths)
        selected_indices = torch.randperm(total_processed_paths)[:num_items]
        for i in selected_indices:
            data, _ = torch.load(self.processed_paths[i])
            # The following commented code block is not needed here:
            # this step is typically performed in the base.py transforms!
            """
            # Shift the center of the scene to be origin (0, 0)
            data.pos[..., :2] -= self.options.max_xy / 2
            """
            data.lidar_center = torch.tensor(self.options.lidar_center, dtype=data.pos.dtype)
            if hasattr(data, 'gt'):
                data.gt = data.gt.to(data.pos.dtype)
            data.point_y = apply_learning_map(
                data.point_y, self.options.learning_map)
            data_list.append(data)
        return self.collate(data_list)

    def get_transform(self):
        if self.mode == "train":
            if hasattr(self.options, "transform"):
                if self.options.transform == "raw":
                    transform_list = []
                else:
                    transform_list = []
            else:
                transform_list = [
                    Center(self.options.max_xy),
                    RandomRotate(degrees=180., axis=2),
                    RandomFlip(0, self.options.max_xy),
                    CenterCrop(self.options.max_xy), 
                    # ZCrop(self.options.max_z),
                    # T.RandomTranslate(self.options.random_jitter)
                ]
            transform = transform_list

        elif self.mode == "val":
            transform = []
        elif self.mode == "test":
            transform = []
        else:
            raise NotImplementedError(
                f"Mode {self.mode} not implemented. Should be in ['train', 'val', 'test']")
        
        transform = T.Compose(transform)
        return transform

    def prepare_test_dataset(self):
        self.items_per_epoch = []
        self.tiles_unique_selection = {}
        self.tiles_min_z = {}
        self.from_idx_to_tile = {}
        idx = 0
        for i in range(self.__superlen__()):
            unique_i, inverse_i = torch.unique((self.__getsuperitem__(
                    i).pos[:, :2] / self.options.max_xy).int(), dim=0, return_inverse=True)
            self.items_per_epoch.append(unique_i.shape[0])
            self.tiles_unique_selection[i] = unique_i
            self.tiles_min_z[i] = torch_scatter.scatter_min(
                    self.__getsuperitem__(i).pos[:, 2], inverse_i, dim=0)[0]
            for j in range(unique_i.shape[0]):
                self.from_idx_to_tile[idx] = (i, unique_i[j])
                idx += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.mode})"

    @property
    def raw_file_names(self):
        laz_dirs = []
        for tile in os.listdir(osp.join(self.options.data_dir, "tiles")):
            tile_dir = osp.join(self.options.data_dir, "tiles", tile)
            if osp.isdir(tile_dir):
                for subtile in os.listdir(tile_dir):
                    if subtile.split(".")[-1] in ["las", "laz"]:
                        laz_dirs.append(osp.join(tile_dir, subtile))
            if tile_dir.split(".")[-1] in ["las", "laz"]:
                laz_dirs.append(tile_dir)

        return laz_dirs

    @property
    def processed_file_names(self):
        if hasattr(self.options, "bulk") and self.options.bulk:
            # If bulk, return all filenames ending with .pt as a list
            return [filename for filename in os.listdir(self.processed_dir) if filename.endswith('.pt')]
        else:
            return [f'grid{1000*self.options.pre_transform_grid_sample:.0f}mm_data.pt']
    
    @property
    def processed_dir(self) -> str:
        # If bulk consider there is also a directory containing val data
        if hasattr(self.options, "bulk") and self.options.bulk:
            return osp.join(self.root, 'processed', self.mode)
        else:
            return osp.join(self.root, 'processed', self.mode if self.mode != "val" else "train")

    """
    def __superlen__(self) -> int:
        return super().__len__()

    def __len__(self) -> int:
        if self.mode != "test":
            return self.items_per_epoch * self.__superlen__()
        else:
            return sum(self.items_per_epoch)
    """
    def __getsuperitem__(self, idx):
        return super().__getitem__(idx)
    
    def __getitem__(self, idx):
        if self.mode != "test":
            if hasattr(self.options, "bulk") and self.options.bulk:
                #item = self.get(idx)
                item = self.__getsuperitem__(idx)
            else:
                item = self.get(idx)
        else:
            this_idx, tile = self.from_idx_to_tile[idx]
            item = self.__getsuperitem__(this_idx)
            keep = (item.pos[:, :2] / self.options.max_xy).int()
            keep = torch.logical_and(
                keep[:, 0] == tile[0], keep[:, 1] == tile[1])
            for k in item.keys:
                setattr(item, k, getattr(item, k)[keep])

            item.pos[:, :2] -= self.options.max_xy * tile
            item.pos[:, -1] -= item.pos[:, -1].min()

            keep = item.pos[:, -1] < self.options.max_z
            for k in item.keys:
                setattr(item, k, getattr(item, k)[keep])

        item.pos_lenght = torch.tensor(item.pos.size(0))
        
        if hasattr(item, 'lidar_center'):
            item.lidar_center = item.lidar_center.unsqueeze(0)
        if hasattr(item, 'gt'):
            item.gt = item.gt.unsqueeze(0)
        
        if self.options.modality == "3D":
            pad = self.options.N_max - item.pos.size(0) if self.mode != "test" else 0
            if self.options.distance == "xyz":
                item.pos_padded = F.pad(item.pos, (0, 0, 0, pad), mode="constant", value=0).unsqueeze(0)
            elif self.options.distance == "xyzk":
                item.pos_padded = F.pad(torch.cat([item.pos, item.intensity], -1), (0, 0, 0, pad), mode="constant", value=0).unsqueeze(0)
            else:
                raise NotImplementedError(
                    f"LiDAR-HD can't produce {self.options.distance}")

        item.features = 2 * torch.cat([item.rgb / 255., item.pos / self.feature_normalizer, item.intensity], -1) - 1
        #del item.intensity

        if self.options.modality == "2D":
            item = self.from3Dto2Ditem(item)

        return item

    def from3Dto2Ditem(self, item):
        del item.pos_lenght
        res, n_dim = self.options.image.res, self.options.image.n_dim
        intensity = item.intensity.squeeze()
        rgb = item.rgb.float()
        labels = item.point_y.float().squeeze()
        xy, z = item.pos[:, :2],  item.pos[:, 2]
        xy = torch.clamp(torch.floor(
                xy / (self.options.max_xy / (res-0.001))), 0, res - 1)
        xy = (xy[:, 0] * res + xy[:, 1]).long()
        features = []
        for values, init in zip([z, intensity, rgb, labels], [0, 0, 0, -1]):
            for mode in ['min', 'max']:
                features.append(gather_values(
                        xy, z, values, mode=mode, res=res, init=init))
        item = torch.cat(features, dim=-1)
        item = item.reshape(res, res, n_dim)
        return item

def gather_values(xy, z, values, mode='max', res=32, init=0):
    img = (torch.ones(
        res*res, values.shape[1] if 1 < len(values.shape) else 1) * init).squeeze()
    z = z.sort(descending=mode == 'max')
    xy, values = xy[z[1]], values[z[1]]
    unique, inverse = torch.unique(xy, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    img.index_put_((xy[perm],), values[perm], accumulate=False)
    img = img.reshape(res, res, -1)
    return img


class OldLidarScenesDataModule(BaseDataModule):
    _DatasetSplit_ = {
        "train": OldLidarScenes,
        "val": OldLidarScenes,
        "test": OldLidarScenes
    }

    def from_labels_to_color(self, labels):
        return from_sem_to_color(apply_learning_map(labels, self.myhparams.learning_map_inv), self.myhparams.color_map)

    def get_feature_names(self):
        return ["red", "green", "blue", "x", "y", "z", "intensity"]

    def describe(self):
        print(self)

        for split in ["train", "val", "test"]:
            if hasattr(self, f"{split}_dataset") and hasattr(getattr(self, f"{split}_dataset"), "data"):
                print(f"{split} data\t", getattr(
                    self, f"{split}_dataset").data)
                if hasattr(getattr(self, f"{split}_dataset").data, "point_y"):
                    for c, n in zip(*np.unique(getattr(self, f"{split}_dataset").data.point_y.flatten().numpy(), return_counts=True)):
                        print(
                            f"class {self.myhparams.raw_class_names[self.myhparams.learning_map_inv[int(c)]]} ({c}) {(20 - len(self.myhparams.raw_class_names[self.myhparams.learning_map_inv[int(c)]]))*' '}  \thas {n} \tpoints")

        if hasattr(self, "val_dataset"):
            lens = [item.pos.size(0) for item in self.val_dataset.items]
            plt.hist(lens)
            plt.title(
                f"size of val dataset items between {min(lens)} and {max(lens)}")
            plt.show()