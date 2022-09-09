import math
import numpy as np
import typing as t

import torch
import torch.nn as nn


class ShapeEval(nn.Module):
    def __init__(self, num_shape, num_param, latent_size=256):
        super(ShapeEval, self).__init__()

        self.num_shape = num_shape
        self.num_param = num_param

        self.num_rot = 1
        self.dim = 2

        self.latent_size = latent_size

        self.fc = nn.Linear(
            self.latent_size * 8,
            self.num_shape * (self.num_param + self.dim + self.num_rot),
        )

        self.param: t.Optional[torch.Tensor] = None
        self.shift: t.Optional[torch.Tensor] = None
        self.rotation: t.Optional[torch.Tensor] = None

    def forward(self, x, pt):
        """
        x: (batch_size, latent_size * 8)
        pt: (num_pt, 2) y, x order
        """
        batch_size = x.shape[0]
        primitives = self.fc(x).reshape(
            batch_size, self.num_shape, -1
        )  # [batch_size, num_shape, num_param + dim + self.num_rot]

        param = primitives[:, :, : self.num_param]  # [batch_size, num_shape, num_param]
        shift = primitives[
            :, :, self.num_param : (self.num_param + self.dim)
        ]  # [batch_size, num_shape, dim]
        rotation = primitives[
            :, :, (self.num_param + self.dim) :
        ]  # [batch_size, num_shape, num_rot]

        self.param = param
        self.shift = shift
        self.rotation = rotation

        pt = self.transform(pt, shift, rotation)
        return self.evalpt(pt, param)

    def transform(self, pt, shift, rotation):
        rotation = rotation.unsqueeze(dim=-2)  # [batch_size, num_shape, 1, num_rot]

        rotation_mat = rotation.new_zeros(
            rotation.shape[:-1] + (2, 2)
        )  # [batch_size, num_shape, 1, 2, 2]
        rotation = rotation[..., 0]  # [batch_size, num_shape, 1]

        rotation_mat[..., 0, 0] = rotation.cos()
        rotation_mat[..., 0, 1] = rotation.sin()
        rotation_mat[..., 1, 0] = -rotation.sin()
        rotation_mat[..., 1, 1] = rotation.cos()

        pt = pt - shift.unsqueeze(dim=-2)  # [batch_size, num_shape, num_pt, dim]
        pt = (rotation_mat * pt.unsqueeze(dim=-1)).sum(
            dim=-2
        )  # [batch_size, num_shape, num_pt, dim]

        return pt

    def shift_vector_prediction(self):
        return self.shift

    def evalpt(self, pt, param):
        return 1


class CircleEval(ShapeEval):
    def __init__(self, num_shape):
        super().__init__(num_shape, 1)

    def evalpt(self, pt, param):
        """
        pt: [batch_size, num_shape, num_pt]
        param: [batch_size, num_shape, num_param]
        """
        dis = pt.norm(dim=-1)
        return dis - param


class SquareEval(ShapeEval):
    def __init__(self, num_shape):
        super().__init__(num_shape, 2)

    def evalpt(self, pt, param):
        """
        pt: [batch_size, num_shape, num_pt]
        param: [batch_size, num_shape, num_param]
        """
        q = pt.abs() - param.unsqueeze(dim=-2)
        dis = q.max(torch.zeros_like(q)).norm(dim=-1)

        q_x = q[..., 0]
        q_y = q[..., 1]

        ret = q_x.max(q_y).min(torch.zeros_like(dis))

        return dis + ret


class MultipleShapesEval(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, x, pt):
        return torch.cat([part(x, pt) for part in self.parts], dim=1)

    def __len__(self):
        return len(self.parts)

    def __iter__(self) -> ShapeEval:
        for part in self.parts:
            yield part


def MultipleShapesEvaluation(num_shape):
    return MultipleShapesEval([SquareEval(num_shape), CircleEval(num_shape)])
