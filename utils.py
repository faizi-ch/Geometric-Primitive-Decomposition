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
    print(data.shape)
    data = data.numpy().transpose((0, 2, 3, 1))
    print(data.shape)

    plt.figure(figsize=(8 * 2, 4 * 2))
    for i in range(16):
        plt.subplot(4, 8, i + 1)
        plt.imshow(data[i], cmap="gray")
        plt.axis("off")
        if titles is not None:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()


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
