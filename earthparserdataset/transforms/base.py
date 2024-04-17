import math
import numbers
import random

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform


class RandomCropSubTileTrain(T.BaseTransform):
    def __init__(self, max_xy) -> None:
        self.max_xy = max_xy
        super().__init__()

    def __call__(self, data):
        xy = data.pos[np.random.randint(data.pos.size(0)), :2]
        while (xy > data.pos_maxi - self.max_xy / 2.).any() or (xy < data.pos[:, :2].min(0)[0] + self.max_xy / 2.).any():
            xy = data.pos[np.random.randint(data.pos.size(0)), :2]
        
        keep = np.abs(data.pos[:, :2] - xy).max(-1)[0] <= self.max_xy / 2.

        del data.pos_maxi

        for k in data.keys:
            setattr(data, k, getattr(data, k)[keep])

        data.pos[..., :2] -= xy
        return data


class RealRandomCropTrain(T.BaseTransform):
    def __init__(self, max_xy, minPts) -> None:
        self.max_xy = max_xy
        self.minPts = minPts
        super().__init__()

    def __call__(self, data):
        low_bound = data.pos[:, :2].min(0)[0] + self.max_xy / 2.
        up_bound = data.pos_maxi - self.max_xy / 2.
        # Randomly select the position of the crop, instead of randomly sampling from the point cloud itself
        xy_arr = np.random.uniform(low=low_bound, high=up_bound)
        xy = torch.from_numpy(xy_arr)
        keep = np.abs(data.pos[:, :2] - xy).max(-1)[0] <= self.max_xy / 2.
        while ((xy > data.pos_maxi - self.max_xy / 2.).any() or
                (xy < data.pos[:, :2].min(0)[0] + self.max_xy / 2.).any() or
                torch.sum(keep == True) < self.minPts): # this last condition checks if there are enough points in the crop
            xy_arr = np.random.uniform(low=low_bound, high=up_bound)
            xy = torch.from_numpy(xy_arr)
            keep = np.abs(data.pos[:, :2] - xy).max(-1)[0] <= self.max_xy / 2.
        
        del data.pos_maxi

        for k in data.keys:
            setattr(data, k, getattr(data, k)[keep])

        data.pos[..., :2] -= xy - self.max_xy / 2.
        return data


class Center(T.BaseTransform):
    def __init__(self, max_xy) -> None:
        self.max_xy = max_xy
        super().__init__()

    def __call__(self, data):
        
        del data.pos_maxi
        
        data.pos[..., :2] -= self.max_xy / 2.
        if hasattr(data, 'gt'):
            data.gt[..., :2] -= self.max_xy / 2.
        if hasattr(data, 'lidar_center'):
            data.lidar_center[..., :2] -= self.max_xy / 2.
        return data


class CenterCrop(T.BaseTransform):
    def __init__(self, max_xy) -> None:
        self.max_xy = max_xy
        super().__init__()

    def __call__(self, data):

        keep = np.abs(data.pos[:, :2]).max(-1)[0] <= self.max_xy / 2.

        exclude_keys = ['gt', 'lidar_center']
        for k in data.keys:
            if k not in exclude_keys:
                setattr(data, k, getattr(data, k)[keep])

        data.pos[..., :2] += self.max_xy / 2.
        if hasattr(data, 'gt'):
            data.gt[..., :2] += self.max_xy / 2.
        if hasattr(data, 'lidar_center'):
            data.lidar_center[..., :2] += self.max_xy / 2.
        return data


class RandomCropSubTileVal(T.BaseTransform):
    def __init__(self, max_xy) -> None:
        self.max_xy = max_xy
        super().__init__()

    def __call__(self, data):
        xy = data.pos[np.random.randint(data.pos.size(0)), :2]
        while (xy > data.pos_maxi - self.max_xy / 2.).any() or (xy < data.pos[:, :2].min(0)[0] + self.max_xy / 2.).any():
            xy = data.pos[np.random.randint(data.pos.size(0)), :2]
        
        keep = np.abs(data.pos[:, :2] - xy).max(-1)[0] <= self.max_xy / 2.

        del data.pos_maxi

        for k in data.keys:
            setattr(data, k, getattr(data, k)[keep])

        data.pos[..., :2] -= xy - self.max_xy / 2.
        return data


class ZCrop(T.BaseTransform):
    def __init__(self, max_z) -> None:
        self.max_z = max_z
        super().__init__()

    def __call__(self, data):
        data.pos[:, -1] -= data.pos[:, -1].min()
        keep = data.pos[:, -1] < self.max_z

        for k in data.keys:
            setattr(data, k, getattr(data, k)[keep])

        return data


class MaxPoints(T.BaseTransform):
    def __init__(self, max_points, seeding=False) -> None:
        self.max_points = max_points
        self.seeding = seeding
        super().__init__()

    def __call__(self, data):

        if data.pos.shape[0] > self.max_points:
            if self.seeding:
                np.random.seed(1234)
            keep = np.random.choice(
                data.pos.shape[0], self.max_points, replace=True)
            exclude_keys = ['gt', 'lidar_center']
            del data.pos_maxi
            for k in data.keys:
                if k not in exclude_keys:
                    setattr(data, k, getattr(data, k)[keep])
        return data


class RandomRotate(T.BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        matrix = torch.tensor(matrix).to(data.pos.device, data.pos.dtype).t()

        data.pos = data.pos @ matrix
        if hasattr(data, 'gt'):
            data.gt[...,:3] = data.gt[...,:3] @ matrix
        if hasattr(data, 'lidar_center'):
            data.lidar_center = data.lidar_center @ matrix

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')


class RandomScale(BaseTransform):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        data.pos_maxi = data.pos_maxi * scale

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'


class RandomFlip(BaseTransform):
    """Flips node positions along a given axis randomly with a given
    probability.

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """

    def __init__(self, axis, max_xy, p=0.5):
        self.axis = axis
        self.p = p
        self.max_xy = max_xy

    def __call__(self, data):
        if random.random() < self.p:
            data.pos[..., self.axis] = - data.pos[..., self.axis]
            if hasattr(data, 'gt'):
                data.gt[..., self.axis] = - data.gt[..., self.axis]
            if hasattr(data, 'lidar_center'):
                data.lidar_center[..., self.axis] = - data.lidar_center[..., self.axis]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'
