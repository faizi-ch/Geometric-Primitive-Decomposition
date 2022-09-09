import math
import numpy as np
from PIL import Image
import cv2
import typing as t
import os
import random
import matplotlib.pyplot as plt

import torch


def set_random_seed(random_seed=None):
    """
    Seed for reproducibility.
    Using random seed for numpy and torch
    """
    if random_seed is None:
        random_seed = 13
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def show_grid(data, titles=None):
    """Make grid."""
    data = data.numpy().transpose((0, 2, 3, 1))

    plt.figure(figsize=(8 * 2, 4 * 2))
    for i in range(16):
        plt.subplot(4, 8, i + 1)
        plt.imshow(data[i], cmap="gray")
        plt.axis("off")
        if titles is not None:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()


def split_dataset(dataset, valid_size=0.2):
    """Split dataset into train and validation set."""

    dataset_len = len(dataset)
    indices = torch.LongTensor(range(dataset_len))
    split = int(math.floor(valid_size * dataset_len))

    # Shuffle the Indices
    indices = indices[torch.randperm(dataset_len)]

    # Split the train and valid set
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


class PILToTensor_for_targets:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # to make a binary mask, set gray(2) to 0 and black and white pet to 1
        target[(target == 1) | (target == 3)] = 1
        target[target == 2] = 0
        # target = scipy.ndimage.median_filter(target, size=(3,3))
        target = target[None, :, :]
        return target


def preprocess_image(
    image: np.ndarray,
    transforms: t.Optional[t.Callable[[np.ndarray], t.Union[torch.Tensor, np.ndarray]]],
    mnist: bool = False,
) -> t.Tuple[torch.Tensor, ...]:
    height, width = image.shape[0], image.shape[1]

    if mnist:
        thresholded = (image > 0).astype(np.uint8)
    else:
        thresholded = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    distances = cv2.distanceTransform(1 - thresholded, cv2.DIST_L2, maskSize=0)
    coords = np.stack(np.meshgrid(range(width), range(height)), axis=-1).reshape(
        (-1, 2)
    )  # -> N, (x, y)
    distances = distances[coords[:, 1], coords[:, 0]]

    dim = max(height, width)

    coords = coords.astype(np.float32)
    coords = (coords + 0.5) / dim - 0.5

    if transforms is not None:
        if not mnist:
            image = Image.fromarray(image, mode="RGB")
        image = transforms(image)
    else:
        image = torch.from_numpy(image).float() / 255
    coords = torch.from_numpy(coords).float()

    distances = (distances <= 0).astype(np.float32)
    return image, coords, distances


def compute_iou(mask2, mask1):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0

    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union


def coverage_threshold(mask2, mask1):
    """
    To compute how much of mask2 (gt) is covered by mask1 (predicted shape)
    """
    intersection = (mask1 * mask2).to(torch.int).count_nonzero()
    if intersection == 0:
        return 0.0

    union = mask1[mask1 == 1].count_nonzero()
    return intersection / union
